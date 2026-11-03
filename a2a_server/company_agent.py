"""公司画像 Agent（端口 5006）。

职责：基于 MySQL 结构化查询，回答"某公司是什么情况"类问题。

风格：LlamaIndex 式的"结构化数据查询"——把自然语言转成 SQL 查询。

能力：
- 公司基本信息（行业、地区、上市日期、全称）
- 公告统计（总量、分层分布、时间跨度）
- 同行业/同地区公司对比

用法：
    python -m a2a_server.company_agent
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a.server.agent_execution import RequestContext

from a2a_server.base import PORTS, BaseExecutor, make_agent_card, run_server
from utils.db import connect
from utils.logger import get_logger

logger = get_logger(__name__)


class CompanyExecutor(BaseExecutor):
    """公司画像 Executor。"""

    agent_name = "company"

    def _find_company(self, query: str) -> dict | None:
        """从查询中识别公司名，返回公司记录。

        ⚠️ 踩坑：不能用 `re.sub` 去掉疑问词后整体匹配，
        因为 "贵州燃气发了多少公告" 去掉词后剩 "贵州燃气发多少"，
        匹配不到 "贵州燃气"。

        正确做法：连续中文段做滑窗子串匹配（同 announcement_agent）。
        """
        conn = connect()
        try:
            with conn.cursor() as cur:
                # 1) 6 位股票代码（最可靠）
                m = re.search(r"\b(\d{6})\b", query)
                if m:
                    cur.execute(
                        "SELECT * FROM companies WHERE sec_code = %s", (m.group(1),)
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)

                # 2) 生成候选：连续中文段的滑窗子串（长度 2-8，从长到短）
                segments = re.findall(r"[\u4e00-\u9fff]+", query)
                candidates: set[str] = set()
                for seg in segments:
                    for length in range(8, 1, -1):
                        for i in range(len(seg) - length + 1):
                            candidates.add(seg[i : i + length])

                ordered = sorted(candidates, key=len, reverse=True)

                # 3) 精确匹配
                for cand in ordered:
                    cur.execute(
                        "SELECT * FROM companies WHERE sec_name = %s "
                        "OR company_abbr = %s LIMIT 1",
                        (cand, cand),
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)

                # 4) 前缀匹配（"贵州" -> "贵州燃气"）
                for cand in ordered:
                    if len(cand) < 2:
                        continue
                    cur.execute(
                        "SELECT * FROM companies WHERE sec_name LIKE %s "
                        "ORDER BY CHAR_LENGTH(sec_name) LIMIT 1",
                        (f"{cand}%",),
                    )
                    row = cur.fetchone()
                    if row:
                        return dict(row)

                return None
        finally:
            conn.close()

    def _get_announcement_stats(self, sec_code: str) -> dict:
        """查某公司的公告统计。"""
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        SUM(level='L1') AS l1,
                        SUM(level='L2') AS l2,
                        SUM(level='L3') AS l3,
                        MIN(pub_date) AS first_date,
                        MAX(pub_date) AS last_date
                    FROM announcements
                    WHERE sec_code = %s
                    """,
                    (sec_code,),
                )
                stats = cur.fetchone() or {}

                # 最近 5 条公告
                cur.execute(
                    """
                    SELECT title, pub_date, level
                    FROM announcements
                    WHERE sec_code = %s
                    ORDER BY pub_date DESC, announcement_time DESC
                    LIMIT 5
                    """,
                    (sec_code,),
                )
                recent = cur.fetchall()

                return {"stats": stats, "recent": recent}
        finally:
            conn.close()

    def _get_peers(self, csrc_code: str | None, area_name: str | None, exclude: str) -> dict:
        """查同行业/同地区公司数量。"""
        if not csrc_code:
            return {}
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS c FROM companies WHERE csrc_code = %s AND sec_code != %s",
                    (csrc_code, exclude),
                )
                industry_count = cur.fetchone()["c"]

                cur.execute(
                    "SELECT COUNT(*) AS c FROM companies WHERE area_name = %s AND sec_code != %s",
                    (area_name, exclude),
                ) if area_name else None
                area_count = cur.fetchone()["c"] if area_name else 0

                return {"industry_peers": industry_count, "area_peers": area_count}
        finally:
            conn.close()

    async def handle(self, query: str, context: RequestContext) -> dict[str, Any]:
        """处理公司查询。"""

        def _do_query():
            company = self._find_company(query)
            if not company:
                return None, None, None
            stats = self._get_announcement_stats(company["sec_code"])
            peers = self._get_peers(
                company.get("csrc_code"), company.get("area_name"), company["sec_code"]
            )
            return company, stats, peers

        company, stats, peers = await asyncio.to_thread(_do_query)

        if not company:
            return {
                "answer": f"没有识别出「{query}」中提到的公司。请提供公司名称或 6 位股票代码。",
                "data": {"query": query, "found": False},
                "sources": [],
            }

        # 构造画像
        lines = [f"【{company['sec_name']}（{company['sec_code']}）】", ""]

        if company.get("full_name"):
            lines.append(f"公司全称：{company['full_name']}")
        if company.get("csrc_desc"):
            lines.append(f"所属行业：{company['csrc_desc']}")
        if company.get("area_desc"):
            lines.append(f"注册地区：{company['area_desc']}")
        if company.get("list_date"):
            lines.append(f"上市日期：{company['list_date']}")
        if company.get("market"):
            market_map = {"SSE": "上海证券交易所", "SZSE": "深圳证券交易所", "BJ": "北京证券交易所"}
            lines.append(f"上市市场：{market_map.get(company['market'], company['market'])}")

        lines.append("")

        s = stats["stats"]
        if s.get("total"):
            lines.append(f"公告统计（{s['first_date']} ~ {s['last_date']}）：")
            lines.append(f"  总计 {s['total']} 条")
            lines.append(f"  核心 L1：{s['l1'] or 0} 条")
            lines.append(f"  重要 L2：{s['l2'] or 0} 条")
            lines.append(f"  模板 L3：{s['l3'] or 0} 条")
            lines.append("")

            if stats["recent"]:
                lines.append("最近公告：")
                for r in stats["recent"]:
                    lines.append(f"  [{r['pub_date']}] {r['title'][:60]}")
        else:
            lines.append("该公司在采集时间范围内没有公告记录。")

        if peers and peers.get("industry_peers"):
            lines.append("")
            lines.append(f"同行业公司：{peers['industry_peers']} 家")

        sources = []
        if company.get("full_name"):
            sources.append(f"公司信息来自上交所公开数据（{company['sec_code']}）")
        if s.get("total"):
            sources.append(f"公告统计来自巨潮资讯，共 {s['total']} 条记录")

        return {
            "answer": "\n".join(lines),
            "data": {
                "query": query,
                "company": {
                    "sec_code": str(company["sec_code"]),
                    "sec_name": str(company["sec_name"]),
                    "full_name": str(company["full_name"]) if company.get("full_name") else None,
                    "csrc_desc": str(company["csrc_desc"]) if company.get("csrc_desc") else None,
                    "area_desc": str(company["area_desc"]) if company.get("area_desc") else None,
                    "list_date": str(company["list_date"]) if company.get("list_date") else None,
                    "market": str(company["market"]) if company.get("market") else None,
                },
                "announcement_stats": {
                    "total": int(s.get("total") or 0),
                    "l1": int(s.get("l1") or 0),
                    "l2": int(s.get("l2") or 0),
                    "l3": int(s.get("l3") or 0),
                    "date_range": f"{s.get('first_date')} ~ {s.get('last_date')}",
                },
                "peers": {
                    "industry_peers": int((peers or {}).get("industry_peers") or 0),
                    "area_peers": int((peers or {}).get("area_peers") or 0),
                },
            },
            "sources": sources,
        }


def create_card(port: int = PORTS["company"]):
    return make_agent_card(
        name="公司画像助手",
        description=(
            "查询上市公司基本信息与公告统计。"
            "支持公司全称、行业、地区、上市日期查询，"
            "以及该公司在采集期内的公告数量和分层分布。"
        ),
        url=f"http://127.0.0.1:{port}/",
        skills=[
            {
                "id": "company_profile",
                "name": "公司基本信息",
                "description": "查询公司的行业、地区、上市日期、全称等",
                "tags": ["公司", "基本信息"],
                "examples": ["介绍一下浦发银行", "600000 是什么公司"],
            },
            {
                "id": "company_announcement_stats",
                "name": "公司公告统计",
                "description": "统计某公司的公告数量、分层分布、最近公告",
                "tags": ["公司", "统计", "公告"],
                "examples": ["贵州燃气发了多少公告", "格科微最近有什么公告"],
            },
        ],
    )


def main() -> None:
    port = PORTS["company"]
    card = create_card(port)
    executor = CompanyExecutor()
    run_server(card, executor, port=port)


if __name__ == "__main__":
    main()
