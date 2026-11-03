"""采集人民币汇率中间价 + 黄金行情。

数据源：
- 中国货币网: chinamoney.com.cn/r/cms/www/chinamoney/data/fx/ccpr.json（25 组货币对）
- 上海黄金交易所: sge.com.cn/graph/Dailyhq?instid=Au99.99

⚠️ 关键坑：SGE 接口必须带 instid 参数，否则返回的 2,392 条记录数值全为 0。

输出：data/raw/fx_rates.json, data/raw/gold_prices.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import RAW_DIR, USER_AGENT
from utils.logger import get_logger, log_banner, log_exception, log_summary

logger = get_logger(__name__)

FX_URL = "https://www.chinamoney.com.cn/r/cms/www/chinamoney/data/fx/ccpr.json"
FX_REFERER = "https://www.chinamoney.com.cn/"

# 必须带 instid，否则返回全 0
GOLD_URL = "https://www.sge.com.cn/graph/Dailyhq?instid=Au99.99"
GOLD_REFERER = "https://www.sge.com.cn/"


def fetch_fx() -> list[dict]:
    """拉取人民币汇率中间价。"""
    headers = {"User-Agent": USER_AGENT, "Referer": FX_REFERER}

    logger.info("请求中国货币网汇率 ...")
    try:
        resp = requests.get(FX_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "汇率请求失败")
        raise

    # 报价日期在 data.lastDate，如 "2026-10-30 9:15"
    last_date_raw = (data.get("data") or {}).get("lastDate") or ""
    rate_date = last_date_raw.split(" ")[0] if last_date_raw else ""
    if not rate_date:
        rate_date = datetime.now().strftime("%Y-%m-%d")
        logger.warning("接口未返回报价日期，使用当前日期 %s", rate_date)

    records = data.get("records") or []
    logger.info("汇率返回 %d 组货币对，报价日期 %s", len(records), rate_date)

    out = []
    skipped = 0
    for r in records:
        price = r.get("price")
        if not price or price in ("---", ""):
            skipped += 1
            continue
        try:
            rate = float(price)
        except (TypeError, ValueError):
            skipped += 1
            continue

        bp = r.get("bpDouble")
        try:
            change_bp = float(bp) if bp not in (None, "---", "") else None
        except (TypeError, ValueError):
            change_bp = None

        out.append(
            {
                "pair": r.get("vrtName") or "",
                "pair_en": r.get("vrtEName") or "",
                "rate": rate,
                "change_bp": change_bp,
                "rate_date": rate_date,
            }
        )

    logger.debug("汇率解析: 有效 %d 条, 跳过 %d 条", len(out), skipped)
    return out


def fetch_gold() -> list[dict]:
    """拉取黄金日行情。

    SGE 返回结构：{"time": [["2016-12-19", o, h, l, c], ...]}
    关键：URL 必须带 ?instid=Au99.99，否则返回全 0（实测踩坑）。
    """
    headers = {"User-Agent": USER_AGENT, "Referer": GOLD_REFERER}

    logger.info("请求上海黄金交易所 ...")
    try:
        resp = requests.get(GOLD_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "黄金请求失败")
        raise

    rows = data.get("time") or []
    logger.info("黄金返回 %d 条原始记录", len(rows))

    out = []
    zero_count = 0
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        d, o, h, l, c = row[0], row[1], row[2], row[3], row[4]
        # 过滤全 0 的空记录
        if not any([o, h, l, c]):
            zero_count += 1
            continue
        out.append(
            {
                "trade_date": d,
                "contract": "Au99.99",
                "open_price": o,
                "high_price": h,
                "low_price": l,
                "close_price": c,
                "volume": None,
            }
        )

    if zero_count:
        logger.warning(
            "黄金数据中有 %d 条全 0 记录被过滤（若全部为 0，检查 URL 是否带 instid 参数）",
            zero_count,
        )

    logger.info("黄金有效记录 %d 条", len(out))
    return out


def main() -> None:
    log_banner(logger, "采集汇率 + 黄金")

    results: dict[str, str] = {}

    # 汇率
    try:
        fx = fetch_fx()
        fp = RAW_DIR / "fx_rates.json"
        fp.write_text(json.dumps(fx, ensure_ascii=False, indent=1), encoding="utf-8")
        results["汇率"] = f"{len(fx)} 条 -> {fp.name}"
        if fx:
            for r in fx[:3]:
                logger.info(
                    "  %-16s %10.4f  变动 %+.0f bp",
                    r["pair"],
                    r["rate"],
                    r["change_bp"] or 0,
                )
    except Exception as e:  # noqa: BLE001
        results["汇率"] = f"失败: {type(e).__name__}"
        logger.error("汇率采集失败: %s", e)

    # 黄金
    try:
        gold = fetch_gold()
        fp = RAW_DIR / "gold_prices.json"
        fp.write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
        results["黄金"] = f"{len(gold):,} 条 -> {fp.name}"
        if gold:
            logger.info(
                "  最新 %s: 收盘 %.2f", gold[-1]["trade_date"], gold[-1]["close_price"]
            )
    except Exception as e:  # noqa: BLE001
        results["黄金"] = f"失败: {type(e).__name__}"
        logger.error("黄金采集失败: %s", e)

    logger.info("")
    log_summary(logger, "采集结果:", results)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "汇率黄金采集异常终止")
        sys.exit(1)
