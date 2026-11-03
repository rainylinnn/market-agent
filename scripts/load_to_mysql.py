"""数据导入器：把 JSON 数据写入 MySQL。

用法：
    python scripts/load_to_mysql.py all            # 全部
    python scripts/load_to_mysql.py announcements  # 只导公告
    python scripts/load_to_mysql.py companies      # 公司信息
    python scripts/load_to_mysql.py fx             # 汇率
    python scripts/load_to_mysql.py gold           # 黄金
    python scripts/load_to_mysql.py quotes         # 行情

设计要点：
- **幂等**：INSERT ... ON DUPLICATE KEY UPDATE，重复运行不产生脏数据
- **分批**：每 1000 条提交一次，避免大事务
- **容错**：单条失败不中断整体，最后汇总失败数
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import RAW_DIR
from utils.db import connect
from utils.logger import get_logger, log_banner, log_exception, log_progress, log_summary

logger = get_logger(__name__)

BATCH_SIZE = 1000


def load_announcements() -> dict:
    """导入所有 data/raw/ann_*.json 到 announcements 表。"""
    files = sorted(RAW_DIR.glob("ann_*.json"))
    if not files:
        logger.warning("没有找到 ann_*.json，请先运行 fetch_announcements.py")
        return {"状态": "跳过", "原因": "无数据文件"}

    logger.info("发现 %d 个日文件", len(files))

    sql = """
    INSERT INTO announcements (
      announcement_id, sec_code, sec_name, title, announcement_time,
      pub_date, plate, adjunct_url, adjunct_size_kb, adjunct_type,
      announcement_type, level
    ) VALUES (
      %(announcement_id)s, %(sec_code)s, %(sec_name)s, %(title)s, %(announcement_time)s,
      %(pub_date)s, %(plate)s, %(adjunct_url)s, %(adjunct_size_kb)s, %(adjunct_type)s,
      %(announcement_type)s, %(level)s
    )
    ON DUPLICATE KEY UPDATE
      title = VALUES(title),
      level = VALUES(level),
      adjunct_url = VALUES(adjunct_url),
      announcement_type = VALUES(announcement_type)
    """

    conn = connect()
    total_ok = 0
    total_err = 0
    batch: list[dict] = []
    failed_files: list[str] = []

    try:
        with conn.cursor() as cur:
            for i, fp in enumerate(files, 1):
                try:
                    records = json.loads(fp.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    logger.warning("读取失败 %s: %s", fp.name, e)
                    failed_files.append(fp.name)
                    continue

                for r in records:
                    if not r.get("announcement_id") or not r.get("sec_code"):
                        continue
                    r.setdefault("plate", None)
                    batch.append(r)

                    if len(batch) >= BATCH_SIZE:
                        try:
                            cur.executemany(sql, batch)
                            conn.commit()
                            total_ok += len(batch)
                        except Exception as e:  # noqa: BLE001
                            conn.rollback()
                            total_err += len(batch)
                            logger.error("批次写入失败 (%d 条): %s", len(batch), e)
                        batch = []

                log_progress(logger, i, len(files), prefix="  公告导入 ")

            # 收尾
            if batch:
                try:
                    cur.executemany(sql, batch)
                    conn.commit()
                    total_ok += len(batch)
                except Exception as e:  # noqa: BLE001
                    conn.rollback()
                    total_err += len(batch)
                    logger.error("最后批次失败 (%d 条): %s", len(batch), e)
    finally:
        conn.close()

    if failed_files:
        logger.warning("有 %d 个文件读取失败: %s", len(failed_files), failed_files[:5])

    logger.info("公告导入完成: 成功 %s 条, 失败 %s 条", f"{total_ok:,}", f"{total_err:,}")
    return {"成功": f"{total_ok:,} 条", "失败": f"{total_err:,} 条"}


def load_companies() -> dict:
    """导入公司基本信息（从 data/raw/companies.json）。"""
    fp = RAW_DIR / "companies.json"
    if not fp.exists():
        logger.warning("找不到 %s，跳过（需先运行 fetch_companies.py）", fp.name)
        return {"状态": "跳过", "原因": "无数据文件"}

    records = json.loads(fp.read_text(encoding="utf-8"))
    logger.info("发现 %s 家公司", f"{len(records):,}")

    sql = """
    INSERT INTO companies (
      sec_code, sec_name, company_abbr, full_name, full_name_en,
      csrc_code, csrc_desc, area_name, area_desc,
      list_board, list_date, delist_date, state_code, product_status, market
    ) VALUES (
      %(sec_code)s, %(sec_name)s, %(company_abbr)s, %(full_name)s, %(full_name_en)s,
      %(csrc_code)s, %(csrc_desc)s, %(area_name)s, %(area_desc)s,
      %(list_board)s, %(list_date)s, %(delist_date)s, %(state_code)s,
      %(product_status)s, %(market)s
    )
    ON DUPLICATE KEY UPDATE
      sec_name = VALUES(sec_name),
      full_name = VALUES(full_name),
      csrc_desc = VALUES(csrc_desc),
      area_desc = VALUES(area_desc),
      market = VALUES(market)
    """

    conn = connect()
    ok = err = 0
    try:
        with conn.cursor() as cur:
            for i, r in enumerate(records, 1):
                if not r.get("sec_code"):
                    continue
                r.setdefault("market", None)
                try:
                    cur.execute(sql, r)
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    err += 1
                    if err <= 3:  # 只打印前几个错误，避免刷屏
                        logger.warning("导入失败 %s: %s", r.get("sec_code"), e)
                log_progress(logger, i, len(records), prefix="  公司导入 ", every=1000)
            conn.commit()
    finally:
        conn.close()

    logger.info("公司信息导入完成: 成功 %s 条, 失败 %s 条", f"{ok:,}", f"{err:,}")
    return {"成功": f"{ok:,} 条", "失败": f"{err:,} 条"}


def load_fx() -> dict:
    """导入汇率（从 data/raw/fx_rates.json）。"""
    fp = RAW_DIR / "fx_rates.json"
    if not fp.exists():
        logger.warning("找不到 %s，跳过（需先运行 fetch_fx_gold.py）", fp.name)
        return {"状态": "跳过", "原因": "无数据文件"}

    payload = json.loads(fp.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records", [])
    else:
        records = []

    logger.info("发现 %d 条汇率", len(records))

    sql = """
    INSERT INTO fx_rates (pair, pair_en, rate, change_bp, rate_date)
    VALUES (%(pair)s, %(pair_en)s, %(rate)s, %(change_bp)s, %(rate_date)s)
    ON DUPLICATE KEY UPDATE
      rate = VALUES(rate),
      change_bp = VALUES(change_bp)
    """

    conn = connect()
    ok = err = 0
    try:
        with conn.cursor() as cur:
            for r in records:
                try:
                    cur.execute(sql, r)
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    err += 1
                    logger.warning("汇率导入失败 %s: %s", r.get("pair"), e)
            conn.commit()
    finally:
        conn.close()

    logger.info("汇率导入完成: 成功 %d 条, 失败 %d 条", ok, err)
    return {"成功": f"{ok} 条", "失败": f"{err} 条"}


def load_gold() -> dict:
    """导入黄金行情（从 data/raw/gold_prices.json）。"""
    fp = RAW_DIR / "gold_prices.json"
    if not fp.exists():
        logger.warning("找不到 %s，跳过（需先运行 fetch_fx_gold.py）", fp.name)
        return {"状态": "跳过", "原因": "无数据文件"}

    records = json.loads(fp.read_text(encoding="utf-8"))
    logger.info("发现 %s 条黄金行情", f"{len(records):,}")

    sql = """
    INSERT INTO gold_prices (
      trade_date, contract, open_price, high_price, low_price, close_price, volume
    ) VALUES (
      %(trade_date)s, %(contract)s, %(open_price)s, %(high_price)s,
      %(low_price)s, %(close_price)s, %(volume)s
    )
    ON DUPLICATE KEY UPDATE
      open_price = VALUES(open_price),
      high_price = VALUES(high_price),
      low_price = VALUES(low_price),
      close_price = VALUES(close_price)
    """

    conn = connect()
    ok = err = 0
    try:
        with conn.cursor() as cur:
            for i, r in enumerate(records, 1):
                try:
                    cur.execute(sql, r)
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    err += 1
                    if err <= 3:
                        logger.warning("黄金导入失败 %s: %s", r.get("trade_date"), e)
                log_progress(logger, i, len(records), prefix="  黄金导入 ", every=500)
            conn.commit()
    finally:
        conn.close()

    logger.info("黄金导入完成: 成功 %s 条, 失败 %s 条", f"{ok:,}", f"{err:,}")
    return {"成功": f"{ok:,} 条", "失败": f"{err:,} 条"}


def load_quotes() -> dict:
    """导入行情快照（从 data/raw/quotes.json）。"""
    fp = RAW_DIR / "quotes.json"
    if not fp.exists():
        logger.warning("找不到 %s，跳过（需先运行 fetch_quotes.py）", fp.name)
        return {"状态": "跳过", "原因": "无数据文件"}

    records = json.loads(fp.read_text(encoding="utf-8"))
    logger.info("发现 %d 条行情", len(records))

    sql = """
    INSERT INTO quotes (
      sec_code, sec_name, price, change_pct, change_amt, market, quote_date, quote_time
    ) VALUES (
      %(sec_code)s, %(sec_name)s, %(price)s, %(change_pct)s,
      %(change_amt)s, %(market)s, %(quote_date)s, %(quote_time)s
    )
    ON DUPLICATE KEY UPDATE
      price = VALUES(price),
      change_pct = VALUES(change_pct),
      change_amt = VALUES(change_amt),
      quote_time = VALUES(quote_time)
    """

    conn = connect()
    ok = err = 0
    try:
        with conn.cursor() as cur:
            for r in records:
                try:
                    cur.execute(sql, r)
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    err += 1
                    logger.warning("行情导入失败 %s: %s", r.get("sec_code"), e)
            conn.commit()
    finally:
        conn.close()

    logger.info("行情导入完成: 成功 %d 条, 失败 %d 条", ok, err)
    return {"成功": f"{ok} 条", "失败": f"{err} 条"}


TASKS = {
    "announcements": ("公告", load_announcements),
    "companies": ("公司信息", load_companies),
    "fx": ("汇率", load_fx),
    "gold": ("黄金", load_gold),
    "quotes": ("行情", load_quotes),
}


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "all"

    if action != "all" and action not in TASKS:
        logger.error("未知的导入目标: %s", action)
        logger.info("可用: all, %s", ", ".join(TASKS.keys()))
        sys.exit(1)

    log_banner(logger, f"数据导入  target={action}")

    targets = TASKS.keys() if action == "all" else [action]
    t0 = datetime.now()
    results: dict[str, dict] = {}

    for idx, key in enumerate(targets, 1):
        name, fn = TASKS[key]
        logger.info("")
        logger.info("[%d/%d] %s", idx, len(targets), name)
        try:
            results[name] = fn()
        except Exception as e:  # noqa: BLE001
            log_exception(logger, e, f"{name} 导入失败")
            results[name] = {"状态": "失败", "错误": type(e).__name__}

    elapsed = (datetime.now() - t0).total_seconds()

    logger.info("")
    log_banner(logger, "导入汇总")
    for name, res in results.items():
        logger.info("  %-8s %s", name, "  ".join(f"{k}={v}" for k, v in res.items()))
    logger.info("  总耗时   %.1f 秒", elapsed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户中断")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "导入任务异常终止")
        sys.exit(1)
