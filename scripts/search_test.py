"""向量检索验证与效果评估。

用法：
    # 交互式检索
    python scripts/search_test.py --query "回购股份"

    # 批量评估（计算 Recall@K）
    python scripts/search_test.py --eval

    # 指定过滤条件
    python scripts/search_test.py --query "减持" --level L1 --top-k 5
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import EMBEDDING
from utils.db import connect
from utils.embedding import get_backend
from utils.logger import get_logger, log_banner, log_exception, log_summary
from utils.milvus_client import get_client, get_stats, search

logger = get_logger(__name__)


def do_search(
    query: str,
    top_k: int = 10,
    level: str | None = None,
    sec_code: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """执行一次检索。"""
    backend = get_backend()
    client = get_client()

    t0 = time.time()
    vec = backend.encode([query])[0]
    embed_ms = (time.time() - t0) * 1000

    t1 = time.time()
    hits = search(
        client,
        vec,
        top_k=top_k,
        level=level,
        sec_code=sec_code,
        date_from=date_from,
        date_to=date_to,
    )
    search_ms = (time.time() - t1) * 1000

    logger.debug("检索耗时: 向量化 %.0fms + 检索 %.0fms", embed_ms, search_ms)
    return hits


def print_hits(hits: list[dict]) -> None:
    """打印检索结果。"""
    if not hits:
        logger.warning("无结果")
        return

    logger.info("")
    logger.info("  %-4s %-8s %-10s %-8s %-12s %s", "排名", "相似度", "代码", "层级", "日期", "标题")
    logger.info("  " + "-" * 100)
    for i, h in enumerate(hits, 1):
        title = (h["title"] or "")[:40]
        logger.info(
            "  %-4d %-8.4f %-10s %-8s %-12s %s",
            i,
            h["score"],
            h["sec_code"] or "-",
            h["level"] or "-",
            h["pub_date"] or "-",
            title,
        )


def keyword_baseline(query: str, top_k: int = 10) -> set[str]:
    """关键词基线：用 SQL LIKE 匹配。

    用于对比向量检索 vs 关键词检索的差异。
    """
    conn = connect()
    try:
        with conn.cursor() as cur:
            # 提取查询中的核心词（简单策略：取最长的连续中文片段）
            keywords = [w for w in query if "\u4e00" <= w <= "\u9fff"]
            core = "".join(keywords[:4]) if keywords else query

            cur.execute(
                """
                SELECT announcement_id FROM announcements
                WHERE title LIKE %s AND level IN ('L1','L2')
                ORDER BY pub_date DESC
                LIMIT %s
                """,
                (f"%{core}%", top_k),
            )
            return {r["announcement_id"] for r in cur.fetchall()}
    finally:
        conn.close()


def evaluate() -> None:
    """批量评估检索效果。

    方法：对每个测试查询，用关键词检索结果作为"伪相关集"，
    计算向量检索的 Recall@K。
    """
    log_banner(logger, "检索效果评估")

    test_queries = [
        "回购股份",
        "股东减持",
        "业绩预告",
        "重大资产重组",
        "股权激励",
        "对外投资",
        "关联交易",
        "诉讼仲裁",
        "利润分配",
        "停牌复牌",
        "控制权变更",
        "破产重整",
    ]

    backend = get_backend()
    client = get_client()

    logger.info("测试查询数: %d", len(test_queries))
    logger.info("")
    logger.info("  %-14s %10s %10s %10s %10s", "查询", "向量耗时", "命中数", "Recall@10", "重叠数")
    logger.info("  " + "-" * 60)

    total_recall = 0.0
    total_latency = 0.0
    valid = 0

    for q in test_queries:
        try:
            t0 = time.time()
            vec = backend.encode([q])[0]
            hits = search(client, vec, top_k=10, level=None)
            latency = (time.time() - t0) * 1000

            if not hits:
                logger.info("  %-14s %10.0fms %10d %10s %10s", q, latency, 0, "N/A", "-")
                continue

            # 关键词基线
            baseline = keyword_baseline(q, top_k=20)
            vec_ids = {h["announcement_id"] for h in hits}
            overlap = len(vec_ids & baseline)

            recall = overlap / min(len(baseline), 10) if baseline else 0.0

            logger.info(
                "  %-14s %10.0fms %10d %9.0f%% %10d",
                q,
                latency,
                len(hits),
                recall * 100,
                overlap,
            )

            total_recall += recall
            total_latency += latency
            valid += 1

        except Exception as e:  # noqa: BLE001
            logger.error("  查询 %s 失败: %s", q, e)

    if valid:
        logger.info("  " + "-" * 60)
        log_summary(
            logger,
            "汇总:",
            {
                "有效查询": valid,
                "平均 Recall@10": f"{total_recall/valid*100:.1f}%",
                "平均延迟": f"{total_latency/valid:.0f}ms",
                "Milvus": get_stats(client),
            },
        )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="向量检索测试")
    p.add_argument("--query", "-q", help="检索词")
    p.add_argument("--top-k", "-k", type=int, default=10, help="返回条数")
    p.add_argument("--level", help="过滤层级 L1/L2/L3")
    p.add_argument("--sec-code", help="过滤股票代码")
    p.add_argument("--date-from", help="起始日期 YYYY-MM-DD")
    p.add_argument("--date-to", help="结束日期 YYYY-MM-DD")
    p.add_argument("--eval", action="store_true", help="批量评估")
    args = p.parse_args()

    try:
        if args.eval:
            evaluate()
            return

        if not args.query:
            logger.error("请用 --query 指定检索词，或用 --eval 批量评估")
            sys.exit(1)

        log_banner(logger, f"检索: {args.query}")

        t0 = time.time()
        hits = do_search(
            args.query,
            top_k=args.top_k,
            level=args.level,
            sec_code=args.sec_code,
            date_from=args.date_from,
            date_to=args.date_to,
        )
        total_ms = (time.time() - t0) * 1000

        print_hits(hits)
        logger.info("")
        logger.info("  共 %d 条结果，总耗时 %.0fms", len(hits), total_ms)

    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "检索失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
