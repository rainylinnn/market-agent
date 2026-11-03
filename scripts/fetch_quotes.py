"""采集实时行情快照（新浪财经）。

数据源：hq.sinajs.cn
- 免费、无需 Key
- 返回 GBK 编码的 JS 变量赋值文本，需解析
- 东方财富接口已对本机 IP 断连，故改用新浪

返回格式：
  var hq_str_sh000001="上证指数,开,昨收,现价,最高,最低,...,日期,时间,00,";

字段索引（指数）：
  0=名称 1=今开 2=昨收 3=现价 4=最高 5=最低 ... 30=日期 31=时间

用法：
  python scripts/fetch_quotes.py                     # 主要指数
  python scripts/fetch_quotes.py --codes 600000,000001
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import RAW_DIR, USER_AGENT
from utils.logger import get_logger, log_banner, log_exception, log_summary

logger = get_logger(__name__)

SINA_URL = "https://hq.sinajs.cn/list={codes}"
SINA_REFERER = "https://finance.sina.com.cn/"

DEFAULT_INDEXES = ["sh000001", "sz399001", "sz399006", "sh000300", "sh000688"]


def to_sina_code(code: str) -> str:
    """6 位代码 -> 新浪代码格式。"""
    code = code.strip()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code.startswith(("6", "5")):
        return f"sh{code}"
    return f"sz{code}"


def fetch_quotes(codes: list[str]) -> list[dict]:
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": SINA_REFERER,
        "Accept": "*/*",
    }
    url = SINA_URL.format(codes=",".join(codes))

    logger.info("请求新浪财经，%d 个标的 ...", len(codes))
    logger.debug("codes=%s", ",".join(codes))

    last_err = None
    text = ""
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            resp.encoding = "gbk"   # 关键：新浪返回 GBK
            text = resp.text
            logger.debug("响应长度 %d 字符", len(text))
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < 3:
                logger.warning("行情请求失败 (attempt=%d/3)，2s 后重试: %s", attempt, e)
                time.sleep(2 * attempt)
            else:
                logger.error("行情请求最终失败: %s", e)
    else:
        raise RuntimeError(f"行情请求失败: {last_err}")

    now = datetime.now()
    quote_date = now.strftime("%Y-%m-%d")
    quote_time = now.strftime("%Y-%m-%d %H:%M:%S")

    out = []
    parse_err = 0
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        try:
            var_part, val_part = line.split("=", 1)
            code = var_part.replace("var hq_str_", "").strip()
            payload = val_part.strip().strip('";')
            if not payload:
                logger.debug("空数据 code=%s", code)
                continue
            fields = payload.split(",")
            if len(fields) < 6:
                logger.warning("字段不足 code=%s 长度=%d", code, len(fields))
                continue

            name = fields[0]
            prev_close = _f(fields[2])
            price = _f(fields[3])

            change_amt = None
            change_pct = None
            if price and prev_close:
                change_amt = round(price - prev_close, 4)
                change_pct = round((price - prev_close) / prev_close * 100, 4)

            out.append(
                {
                    "sec_code": code,
                    "sec_name": name,
                    "price": price,
                    "change_pct": change_pct,
                    "change_amt": change_amt,
                    "market": "sh" if code.startswith("sh") else "sz",
                    "quote_date": quote_date,
                    "quote_time": quote_time,
                }
            )
        except Exception as e:  # noqa: BLE001
            parse_err += 1
            logger.debug("解析失败: %s", e)
            continue

    logger.info("解析出 %d 条（失败 %d 条）", len(out), parse_err)
    return out


def _f(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="采集实时行情（新浪）")
    p.add_argument("--codes", help="逗号分隔的 6 位代码，如 600000,000001")
    args = p.parse_args()

    log_banner(logger, "采集实时行情（新浪财经）")

    if args.codes:
        targets = [to_sina_code(c) for c in args.codes.split(",") if c.strip()]
    else:
        targets = DEFAULT_INDEXES

    try:
        quotes = fetch_quotes(targets)
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "行情采集失败")
        sys.exit(1)

    fp = RAW_DIR / "quotes.json"
    fp.write_text(json.dumps(quotes, ensure_ascii=False, indent=1), encoding="utf-8")

    logger.info("")
    for q in quotes:
        pct = q["change_pct"]
        arrow = "↑" if pct and pct > 0 else ("↓" if pct and pct < 0 else "—")
        pct_s = f"{pct:+.2f}%" if pct is not None else "N/A"
        logger.info(
            "  %-10s %12s  %s %9s",
            q["sec_name"],
            f"{q['price']:,.2f}" if q["price"] else "N/A",
            arrow,
            pct_s,
        )

    log_summary(logger, "", {"行情条数": len(quotes), "输出文件": str(fp)})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "行情采集异常终止")
        sys.exit(1)
