"""公告向量化：MySQL -> 千问/BGE-M3 -> Milvus。

流程：
1. 从 MySQL 读取未向量化的公告（is_embedded=0）
2. 按层过滤（默认 L1+L2，跳过 L3 模板）
3. 批量调用 embedding 后端
4. 写入 Milvus
5. 回写 MySQL 的 is_embedded 标记

特性：
- **断点续传**：每批处理完立即更新标记，中断后继续
- **并发控制**：可配置并发数
- **进度显示**：每 N 条打印进度和速度
- **失败重试**：单批失败不影响整体

用法：
    # 先验证 100 条
    python scripts/embed_announcements.py --limit 100

    # 全量（L1+L2）
    python scripts/embed_announcements.py

    # 只做 L1
    python scripts/embed_announcements.py --level L1

    # 重跑（忽略已标记）
    python scripts/embed_announcements.py --force

    # 只统计不执行
    python scripts/embed_announcements.py --dry-run
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import EMBEDDING, EMBEDDING_DIM, MILVUS
from utils.db import connect
from utils.embedding import QuotaExhaustedError, get_backend
from utils.logger import (
    get_logger,
    log_banner,
    log_exception,
    log_progress,
    log_summary,
)
from utils.milvus_client import (
    ensure_collection,
    flush,
    get_client,
    get_stats,
    insert_batch,
)

logger = get_logger(__name__)

# 每批处理条数（受 embedding 后端批量上限约束）
MILVUS_BATCH = 200


def fetch_pending(
    levels: Sequence[str],
    limit: int | None = None,
    force: bool = False,
) -> list[dict]:
    """从 MySQL 读取待向量化的公告。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(levels))
            where = f"level IN ({placeholders})"
            if not force:
                where += " AND is_embedded = 0"

            sql = f"""
                SELECT announcement_id, sec_code, sec_name, title, level, pub_date
                FROM announcements
                WHERE {where}
                ORDER BY pub_date DESC, announcement_id
            """
            params: list = list(levels)
            if limit:
                sql += " LIMIT %s"
                params.append(limit)

            cur.execute(sql, params)
            rows = cur.fetchall()
            logger.info("从 MySQL 读取 %s 条待向量化记录", f"{len(rows):,}")
            return list(rows)
    finally:
        conn.close()


def count_by_level() -> dict[str, int]:
    """统计各层公告数量（含已向量化状态）。"""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT level,
                       COUNT(*) AS total,
                       SUM(is_embedded = 1) AS embedded,
                       SUM(is_embedded = 0) AS pending
                FROM announcements
                GROUP BY level
                ORDER BY level
            """)
            return {r["level"]: r for r in cur.fetchall()}
    finally:
        conn.close()


def mark_embedded(ids: Sequence[str]) -> int:
    """标记为已向量化。"""
    if not ids:
        return 0
    conn = connect()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(
                f"UPDATE announcements SET is_embedded = 1 WHERE announcement_id IN ({placeholders})",
                list(ids),
            )
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


def build_text(row: dict) -> str:
    """构造用于向量化的文本。

    ⚠️ 实测结论（scripts/compare_embedding.py）：
        只编码标题           -> 平均相似度 0.7140
        公司名 + 标题        -> 平均相似度 0.5949（降 17%）

    原因：公司名与查询意图无关，属于噪音，会稀释语义信号。
    因此这里只编码标题。

    公司名单独存为 Milvus 标量字段 sec_name，
    需要按公司过滤时用表达式 `sec_name like "浦发%"`。
    """
    return (row.get("title") or "").strip()


def run(
    levels: Sequence[str] = ("L1", "L2"),
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """执行向量化。"""
    log_banner(logger, "公告向量化  MySQL -> Milvus")

    # ---------- 1. 统计现状 ----------
    stats = count_by_level()
    logger.info("当前数据状态:")
    logger.info("  %-6s %10s %12s %12s", "层级", "总数", "已向量化", "待处理")
    logger.info("  " + "-" * 44)
    for lv in ("L1", "L2", "L3"):
        if lv in stats:
            s = stats[lv]
            logger.info(
                "  %-6s %10s %12s %12s",
                lv,
                f"{s['total']:,}",
                f"{s['embedded']:,}",
                f"{s['pending']:,}",
            )

    total_pending = sum(stats[lv]["pending"] for lv in levels if lv in stats)
    logger.info("  " + "-" * 44)
    logger.info("  待处理合计（%s）: %s 条", "+".join(levels), f"{total_pending:,}")

    if dry_run:
        logger.info("")
        logger.info("--dry-run 模式，不执行实际向量化")
        return

    if total_pending == 0 and not force:
        logger.info("")
        logger.info("没有待处理的公告，退出")
        return

    # ---------- 2. 读取数据 ----------
    rows = fetch_pending(levels, limit=limit, force=force)
    if not rows:
        logger.warning("没有读到数据，退出")
        return

    # ---------- 3. 初始化后端 ----------
    logger.info("")
    logger.info("初始化向量化后端 ...")
    backend = get_backend()
    logger.info(
        "  后端=%s 模型=%s 维度=%d 批量上限=%d",
        backend.name,
        EMBEDDING["model"] if backend.name == "qwen" else EMBEDDING["bge_model_path"],
        backend.dim,
        backend.max_batch,
    )

    if backend.dim != EMBEDDING_DIM:
        logger.error(
            "维度不匹配！后端=%d Milvus集合=%d", backend.dim, EMBEDDING_DIM
        )
        raise RuntimeError("向量维度与 Milvus 集合不一致，请检查 EMBEDDING_DIM 配置")

    # ---------- 4. 初始化 Milvus ----------
    logger.info("")
    logger.info("初始化 Milvus 集合 ...")
    client = get_client()
    ensure_collection(client)
    logger.info("  %s", get_stats(client))

    # ---------- 5. 批量处理 ----------
    logger.info("")
    log_banner(logger, f"开始向量化  共 {len(rows):,} 条")

    t_start = time.time()
    total_ok = 0
    total_fail = 0
    batch_count = 0
    fail_streak = 0  # 连续失败批次数，成功一批就清零

    for start in range(0, len(rows), MILVUS_BATCH):
        batch = rows[start : start + MILVUS_BATCH]
        batch_count += 1
        t0 = time.time()

        try:
            # 构造文本
            texts = [build_text(r) for r in batch]

            # 向量化
            vectors = backend.encode(texts, batch_size=EMBEDDING["batch_size"])

            if len(vectors) != len(batch):
                logger.error(
                    "向量数不符 期望=%d 实际=%d，跳过本批", len(batch), len(vectors)
                )
                total_fail += len(batch)
                continue

            # 组装 Milvus 记录
            milvus_rows = []
            for r, vec in zip(batch, vectors):
                milvus_rows.append(
                    {
                        "announcement_id": r["announcement_id"],
                        "sec_code": r["sec_code"],
                        "sec_name": r["sec_name"] or "",
                        "title": (r["title"] or "")[:1000],
                        "level": r["level"],
                        "pub_date": str(r["pub_date"]),
                        "embedding": vec,
                    }
                )

            # 写入 Milvus
            inserted = insert_batch(client, milvus_rows)

            # 回写 MySQL 标记
            ids = [r["announcement_id"] for r in batch]
            marked = mark_embedded(ids)

            total_ok += len(batch)
            dt = time.time() - t0
            speed = len(batch) / dt if dt > 0 else 0

            elapsed = time.time() - t_start
            done = total_ok + total_fail
            eta = (elapsed / done) * (len(rows) - done) if done else 0

            logger.info(
                "  批次 %3d  已处理 %6s/%s  插入 %3d  标记 %3d  %5.1fs  %5.1f 条/秒  剩余 %.0f 分钟",
                batch_count,
                f"{done:,}",
                f"{len(rows):,}",
                inserted,
                marked,
                dt,
                speed,
                eta / 60,
            )

        except QuotaExhaustedError as e:
            logger.error("")
            logger.error("=" * 70)
            logger.error("  停止任务：%s", e)
            logger.error("  已完成 %s 条，剩余 %s 条未处理", f"{total_ok:,}", f"{len(rows)-total_ok-total_fail:,}")
            logger.error("  切换到其他模型后可续跑（断点续传会自动跳过已完成）:")
            logger.error("    $env:EMBEDDING_MODEL='qwen3.7-text-embedding'")
            logger.error("    python scripts/embed_announcements.py --level L1,L2")
            logger.error("=" * 70)
            break

        except Exception as e:  # noqa: BLE001
            log_exception(logger, e, f"批次 {batch_count} 失败（{len(batch)} 条）")
            total_fail += len(batch)

            # 连续失败 5 批就停，避免空转刷日志
            fail_streak += 1
            if fail_streak >= 5:
                logger.error(
                    "连续 %d 批失败，停止任务（检查网络、Milvus 或 API 额度）", fail_streak
                )
                break
            continue

        # 成功一批就重置连续失败计数
        fail_streak = 0

    # ---------- 6. 收尾 ----------
    logger.info("")
    logger.info("flush 到磁盘 ...")
    flush(client)

    elapsed_min = (time.time() - t_start) / 60

    logger.info("")
    log_banner(logger, "向量化完成")
    log_summary(
        logger,
        "统计:",
        {
            "成功": f"{total_ok:,} 条",
            "失败": f"{total_fail:,} 条",
            "总耗时": f"{elapsed_min:.1f} 分钟",
            "平均速度": f"{total_ok/max(elapsed_min,0.01):,.0f} 条/分钟",
            "Milvus集合": MILVUS["collection"],
            **{k: v for k, v in get_stats(client).items() if k != "集合"},
        },
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="公告向量化")
    p.add_argument(
        "--level",
        default="L1,L2",
        help="要向量化的层级，逗号分隔（默认 L1,L2）",
    )
    p.add_argument("--limit", type=int, help="最多处理多少条（用于测试）")
    p.add_argument("--force", action="store_true", help="忽略已向量化标记，全部重跑")
    p.add_argument("--dry-run", action="store_true", help="只统计不执行")
    args = p.parse_args()

    levels = [x.strip().upper() for x in args.level.split(",") if x.strip()]

    try:
        run(levels, limit=args.limit, force=args.force, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.warning("用户中断。已处理的数据已保存，重新运行会跳过")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "向量化任务异常终止")
        sys.exit(1)


if __name__ == "__main__":
    main()
