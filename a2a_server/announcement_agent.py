"""公告检索 Agent（端口 5005）。

职责：基于 Milvus 向量检索，回答"某类/某公司公告"类问题。

风格：CrewAI 式的角色化设计——扮演"投研助理"，专注公告检索。

能力：
- 语义检索（向量相似度）
- 按公司/层级/日期过滤
- 返回带出处的结果（announcement_id 可上巨潮核对）

用法：
    python -m a2a_server.announcement_agent
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a.server.agent_execution import RequestContext

from a2a_server.base import (
    PORTS,
    BaseExecutor,
    make_agent_card,
    run_server,
)
from utils.config import MILVUS
from utils.db import connect
from utils.embedding import get_backend
from utils.logger import get_logger
from utils.milvus_client import get_client, search
from utils.rerank import rerank_announcements

logger = get_logger(__name__)

# 两阶段检索参数
RECALL_TOP_K = 50   # 向量召回数量
RERANK_TOP_N = 10   # rerank 后返回数量


class AnnouncementExecutor(BaseExecutor):
    """公告检索 Executor。"""

    agent_name = "announcement"

    def __init__(self) -> None:
        super().__init__()
        self._backend = None
        self._client = None

    def _ensure_init(self) -> None:
        """懒加载向量化后端和 Milvus 客户端（避免启动时阻塞）。"""
        if self._backend is None:
            self.logger.info("初始化向量化后端 ...")
            self._backend = get_backend()
            self.logger.info(
                "  后端=%s dim=%d", self._backend.name, self._backend.dim
            )
        if self._client is None:
            self.logger.info("初始化 Milvus 客户端 ...")
            self._client = get_client()

    def _parse_query(self, query: str) -> dict[str, Any]:
        """从自然语言中解析检索参数。

        用简单的规则匹配，不调 LLM（快且稳定）。
        复杂解析交给 orchestrator 层。
        """
        import re

        params: dict[str, Any] = {
            "top_k": 10,
            "level": None,
            "sec_name": None,
            "sec_code": None,
            "date_from": None,
            "date_to": None,
        }

        # ---- 公司识别（最重要，避免"浦发银行"被当成语义检索词）----
        company = self._detect_company(query)
        if company:
            params["sec_code"] = company["sec_code"]
            params["sec_name"] = company["sec_name"]
            self.logger.debug("识别到公司: %s(%s)", company["sec_name"], company["sec_code"])

        # ---- 层级 ----
        if "核心" in query or "重要公告" in query:
            params["level"] = "L1"

        # ---- 日期 ----
        m = re.search(r"最近\s*(\d+)\s*个?月", query)
        if m:
            from datetime import date, timedelta

            months = int(m.group(1))
            d = date.today()
            params["date_from"] = (d - timedelta(days=months * 30)).isoformat()
            params["date_to"] = d.isoformat()
        elif "最近一个月" in query or "近一个月" in query:
            from datetime import date, timedelta

            d = date.today()
            params["date_from"] = (d - timedelta(days=30)).isoformat()
            params["date_to"] = d.isoformat()

        # ---- 数量 ----
        m = re.search(r"(\d+)\s*条", query)
        if m:
            params["top_k"] = min(int(m.group(1)), 50)

        return params

    def _detect_company(self, query: str) -> dict | None:
        """从查询中识别公司（股票代码 > 精确名 > 前缀 > 滑窗子串）。

        ⚠️ 踩坑：不能用 `re.findall(r'[\\u4e00-\\u9fff]{2,8}')` 抽候选，
        因为正则是贪婪的，"浦发银行最近有什么公告" 会整体匹配成
        "浦发银行最近有什"，切不出"浦发银行"。

        正确做法：把所有连续中文段，按 2-6 字做**滑窗**，从长到短匹配。
        """
        import re

        conn = connect()
        try:
            with conn.cursor() as cur:
                # 1) 6 位股票代码（最可靠）
                m = re.search(r"\b(\d{6})\b", query)
                if m:
                    cur.execute(
                        "SELECT sec_code, sec_name FROM companies WHERE sec_code = %s",
                        (m.group(1),),
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)

                # 2) 生成候选：连续中文段的所有滑窗子串（长度 2-6）
                segments = re.findall(r"[\u4e00-\u9fff]+", query)
                candidates: set[str] = set()
                for seg in segments:
                    for length in range(6, 1, -1):
                        for i in range(len(seg) - length + 1):
                            candidates.add(seg[i : i + length])

                # 按长度降序（长片段更可能是完整公司名）
                ordered = sorted(candidates, key=len, reverse=True)

                # 3) 精确匹配
                for cand in ordered:
                    cur.execute(
                        "SELECT sec_code, sec_name FROM companies WHERE sec_name = %s LIMIT 1",
                        (cand,),
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)

                # 4) 前缀匹配（"浦发" -> "浦发银行"）
                for cand in ordered:
                    if len(cand) < 2:
                        continue
                    cur.execute(
                        "SELECT sec_code, sec_name FROM companies "
                        "WHERE sec_name LIKE %s ORDER BY CHAR_LENGTH(sec_name) LIMIT 1",
                        (f"{cand}%",),
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)

                return None
        finally:
            conn.close()

    async def handle(self, query: str, context: RequestContext) -> dict[str, Any]:
        """处理检索请求。"""
        self._ensure_init()

        params = self._parse_query(query)

        # 构造检索用的语义文本：
        # 如果识别到公司，去掉公司名，只保留意图部分（"最近有什么公告"）
        # 否则 "浦发银行最近有什么公告" 的向量会偏向"银行"语义
        if params["sec_name"]:
            search_text = query.replace(params["sec_name"], "").strip()
            # 去掉残留的疑问词
            for w in ["的", "了", "吗", "呢", "有哪些", "有什么", "最近", "公告", "请", "帮我", "查一下", "看看"]:
                search_text = search_text.replace(w, "")
            search_text = search_text.strip()
            if len(search_text) < 2:
                search_text = "公告"
            self.logger.debug(
                "公司过滤: %s(%s)  检索词: %r",
                params["sec_name"],
                params["sec_code"],
                search_text,
            )
        else:
            search_text = query

        def _do_search():
            # ----- 阶段 1：向量召回（宽召回）-----
            vec = self._backend.encode([search_text])[0]
            recall_k = max(RECALL_TOP_K, params["top_k"] * 3)
            hits = search(
                self._client,
                vec,
                top_k=recall_k,
                level=params["level"],
                sec_code=params["sec_code"],
                date_from=params["date_from"],
                date_to=params["date_to"],
            )
            self.logger.debug("向量召回 %d 条", len(hits))

            if not hits:
                return hits, None

            # ----- 阶段 2：rerank 精排 -----
            # 用原始 query 做 rerank（不是去掉公司名的 search_text），
            # 因为 rerank 能理解完整语义
            top_n = min(params["top_k"], len(hits))
            reranked = rerank_announcements(
                query,
                hits,
                top_n=top_n,
                text_field="title",
            )
            self.logger.debug("rerank 后 %d 条", len(reranked))
            return reranked, len(hits)

        hits, recall_count = await asyncio.to_thread(_do_search)

        if not hits:
            return {
                "answer": f"没有检索到与「{query}」相关的公告。",
                "data": {"query": query, "params": params, "count": 0},
                "sources": [],
            }

        # 构造回答
        lines = [f"检索到 {len(hits)} 条相关公告：", ""]
        for i, h in enumerate(hits, 1):
            lines.append(
                f"{i}. [{h['pub_date']}] {h['sec_name']}（{h['sec_code']}）"
                f"—— {h['title']}"
            )

        # 构造来源（可核对）
        sources = [
            f"{h['sec_name']} {h['title']}（公告ID: {h['announcement_id']}，"
            f"相关度 {h.get('rerank_score', h['score']):.3f}）"
            for h in hits
        ]

        return {
            "answer": "\n".join(lines),
            "data": {
                "query": query,
                "params": params,
                "recall_count": recall_count,
                "count": len(hits),
                "hits": [
                    {
                        "announcement_id": h["announcement_id"],
                        "sec_code": h["sec_code"],
                        "sec_name": h["sec_name"],
                        "title": h["title"],
                        "level": h["level"],
                        "pub_date": h["pub_date"],
                        "rerank_score": h.get("rerank_score"),
                        "vector_score": h.get("vector_score", h.get("score")),
                    }
                    for h in hits
                ],
            },
            "sources": sources,
        }


def create_card(port: int = PORTS["announcement"]):
    return make_agent_card(
        name="公告检索助手",
        description=(
            "基于语义向量检索的上市公司公告查询助手。"
            "可以查找回购、重组、减持、业绩预告等各类公告，"
            "支持按公司、时间范围、重要性过滤。"
        ),
        url=f"http://127.0.0.1:{port}/",
        skills=[
            {
                "id": "search_announcement",
                "name": "公告语义检索",
                "description": "用自然语言描述想找的公告，返回相关公告列表",
                "tags": ["公告", "检索", "向量搜索"],
                "examples": [
                    "最近有哪些公司回购股份",
                    "查找重大资产重组相关公告",
                    "浦发银行最近的公告",
                ],
            },
            {
                "id": "search_by_company",
                "name": "按公司查公告",
                "description": "指定公司名称，返回该公司发布的公告",
                "tags": ["公告", "公司"],
                "examples": ["格科微的回购公告", "查一下贵州燃气的公告"],
            },
        ],
    )


def main() -> None:
    port = PORTS["announcement"]
    card = create_card(port)
    executor = AnnouncementExecutor()
    run_server(card, executor, port=port)


if __name__ == "__main__":
    main()
