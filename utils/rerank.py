"""Rerank 精排模块（千问 qwen3.7-text-rerank）。

职责：对向量召回的候选集做二次排序，提升准确率。

为什么需要 rerank：
    向量检索（embedding）是双塔模型，query 和 doc 分别编码后算余弦相似度，
    精度有限。典型相似度分布是 0.55-0.86，区分度不高。

    rerank 是交叉编码器，把 query 和 doc 拼在一起送入模型，
    能捕捉细粒度语义关系。实测分数区分度大得多：

        query: "回购股份"
        "关于回购公司股份的公告"  → 0.6322
        "关于董事会换届的公告"    → 0.1591      ← 差 4 倍

标准架构：
    向量召回 Top 50  →  rerank 精排  →  返回 Top 10

接口（注意：不是 OpenAI 兼容路径）：
    POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank

用法：
    from utils.rerank import rerank

    docs = ["公告1", "公告2", ...]
    ranked = rerank("回购股份", docs, top_n=10)
    for r in ranked:
        print(r["index"], r["score"], docs[r["index"]])
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import EMBEDDING
from utils.logger import get_logger

logger = get_logger(__name__)

# DashScope 原生接口（不是 OpenAI 兼容路径）
RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)

DEFAULT_MODEL = "qwen3.7-text-rerank"
# 单次最多 500 篇文档
MAX_DOCS = 500
DEFAULT_RETRIES = 3


class RerankError(RuntimeError):
    """Rerank 调用异常。"""


class QuotaExhaustedError(RerankError):
    """额度耗尽。"""


def rerank(
    query: str,
    documents: Sequence[str],
    *,
    top_n: int | None = None,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """对候选文档重新排序。

    参数：
        query: 用户查询
        documents: 候选文档列表（通常来自向量召回）
        top_n: 只返回前 N 个（默认全部）
        model: 模型名
        api_key: 覆盖配置中的 key

    返回：
        [{"index": 原始下标, "score": 相关性分数, "text": 文档内容}, ...]
        按 score 降序

    异常：
        QuotaExhaustedError: 额度用尽
        RerankError: 其他失败
    """
    if not documents:
        return []

    key = api_key or EMBEDDING.get("api_key", "")
    if not key:
        raise RerankError("缺少 QIANWEN_API_KEY，无法调用 rerank")

    # 截断过长文档（rerank 对 token 有限制）
    docs = [(d or "").strip()[:4000] or "（空）" for d in documents]
    if len(docs) > MAX_DOCS:
        logger.warning("文档数 %d 超过上限 %d，截断", len(docs), MAX_DOCS)
        docs = docs[:MAX_DOCS]

    payload = {
        "model": model,
        "input": {"query": query, "documents": docs},
        "parameters": {
            "return_documents": False,
            "top_n": min(top_n or len(docs), len(docs)),
        },
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    logger.debug("rerank 请求 model=%s query=%r docs=%d", model, query[:40], len(docs))

    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            resp = requests.post(
                RERANK_URL, headers=headers, json=payload, timeout=timeout
            )
            elapsed = (time.time() - t0) * 1000

            if resp.status_code == 429:
                wait = 5 * attempt
                logger.warning("rerank 限流 429，等待 %ds (%d/%d)", wait, attempt, retries)
                time.sleep(wait)
                continue

            body = resp.text[:500]

            if resp.status_code != 200:
                if "Free quota exhausted" in body or "AllocationQuota" in body:
                    raise QuotaExhaustedError(f"rerank 额度已用尽（{model}）")

                logger.error("rerank HTTP %d: %s", resp.status_code, body)
                if 400 <= resp.status_code < 500:
                    raise RerankError(f"rerank 失败 HTTP {resp.status_code}: {body}")
                last_err = RerankError(f"HTTP {resp.status_code}")
                time.sleep(2**attempt)
                continue

            data = resp.json()
            output = (data.get("output") or {}).get("results") or []

            results = []
            for item in output:
                idx = item.get("index")
                score = item.get("relevance_score", 0.0)
                results.append(
                    {
                        "index": idx,
                        "score": round(float(score), 4),
                        "text": docs[idx] if idx is not None and idx < len(docs) else "",
                    }
                )

            results.sort(key=lambda x: -x["score"])

            usage = data.get("usage") or {}
            logger.debug(
                "rerank 完成 耗时=%.0fms 返回=%d 篇 tokens=%s",
                elapsed,
                len(results),
                usage.get("total_tokens"),
            )

            return results

        except (QuotaExhaustedError, RerankError):
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                logger.warning(
                    "rerank 失败 (%d/%d)，%ds 后重试: %s", attempt, retries, 2**attempt, e
                )
                time.sleep(2**attempt)
            else:
                logger.error("rerank 最终失败: %s", e)

    raise RerankError(f"rerank 失败（已重试 {retries} 次）: {last_err}")


def rerank_announcements(
    query: str,
    hits: Sequence[dict],
    *,
    top_n: int = 10,
    text_field: str = "title",
    score_threshold: float | None = None,
) -> list[dict]:
    """对向量检索结果做 rerank。

    参数：
        query: 用户查询
        hits: 向量检索返回的命中列表（每项需含 text_field 指定的字段）
        top_n: 返回条数
        text_field: 用于 rerank 的文本字段名
        score_threshold: 分数下限（过滤明显不相关的）

    返回：
        重新排序后的命中列表，附加 `rerank_score` 字段
    """
    if not hits:
        return []

    docs = [h.get(text_field, "") for h in hits]

    try:
        ranked = rerank(query, docs, top_n=top_n)
    except QuotaExhaustedError:
        logger.warning("rerank 额度不足，回退到原始向量排序")
        return list(hits[:top_n])
    except Exception as e:  # noqa: BLE001
        logger.warning("rerank 失败，回退到原始向量排序: %s", e)
        return list(hits[:top_n])

    out = []
    for r in ranked:
        idx = r["index"]
        if idx is None or idx >= len(hits):
            continue
        if score_threshold is not None and r["score"] < score_threshold:
            continue

        item = dict(hits[idx])
        item["rerank_score"] = r["score"]
        item["vector_score"] = item.get("score")  # 保留原始向量分数
        out.append(item)

    logger.debug("rerank 完成: %d -> %d 条", len(hits), len(out))
    return out


def health_check() -> bool:
    """健康检查。"""
    try:
        t0 = time.time()
        ranked = rerank(
            "公司回购股份",
            [
                "关于回购公司股份的公告",
                "关于董事会换届选举的公告",
                "关于2026年第三季度报告披露的提示性公告",
            ],
            top_n=3,
        )
        elapsed = (time.time() - t0) * 1000

        logger.info("rerank 健康检查通过 耗时=%.0fms", elapsed)
        for r in ranked:
            logger.info("  %.4f  %s", r["score"], r["text"])

        # 验证：回购相关应该排第一
        if ranked and "回购" in ranked[0]["text"]:
            logger.info("  排序正确")
            return True

        logger.warning("  排序异常：回购相关未排第一")
        return False

    except Exception as e:  # noqa: BLE001
        logger.error("rerank 健康检查失败: %s", e)
        return False


if __name__ == "__main__":
    from utils.logger import log_banner

    log_banner(logger, "Rerank 模块自测")
    ok = health_check()
    sys.exit(0 if ok else 1)
