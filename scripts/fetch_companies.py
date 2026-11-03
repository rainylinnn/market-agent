"""采集上市公司基本信息（上交所 + 深交所/北交所）。

数据源：
- 上交所: query.sse.com.cn（需 Referer，返回 JSON）
- 深市/北交所: 从公告数据反推（这些交易所没有稳定的公开列表接口）

输出：data/raw/companies.json

日志：
- 请求细节在 DEBUG
- 采集结果在 INFO
- 接口失败在 WARNING（有兜底方案，不中断）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import RAW_DIR, USER_AGENT
from utils.logger import get_logger, log_banner, log_exception, log_summary

logger = get_logger(__name__)

SSE_URL = (
    "https://query.sse.com.cn/sseQuery/commonQuery.do"
    "?sqlId=COMMON_SSE_CP_GPJCTPZ_GPLB_GP_L"
    "&type=inParams&STOCK_TYPE=1&REG_PROVINCE=&CSRC_CODE=&STOCK_CODE="
    "&COMPANY_STATUS=2,4,5,7,8&_={ts}"
)
SSE_REFERER = "https://www.sse.com.cn/"


def fetch_sse() -> list[dict]:
    """拉取上交所上市公司列表。"""
    ts = int(time.time() * 1000)
    url = SSE_URL.format(ts=ts)
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": SSE_REFERER,
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    logger.info("请求上交所上市公司列表 ...")
    logger.debug("URL=%s", url.split("&_=")[0])

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "上交所请求失败")
        raise

    rows = data.get("result") or data.get("pageHelp", {}).get("data") or []
    logger.info("上交所返回 %d 条", len(rows))

    out = []
    skipped = 0
    for r in rows:
        code = r.get("A_STOCK_CODE") or ""
        if not code or code == "-":
            skipped += 1
            continue

        out.append(
            {
                "sec_code": code,
                "sec_name": r.get("SEC_NAME_CN") or r.get("COMPANY_ABBR") or "",
                "company_abbr": r.get("COMPANY_ABBR"),
                "full_name": r.get("FULL_NAME"),
                "full_name_en": r.get("FULL_NAME_IN_ENGLISH"),
                "csrc_code": r.get("CSRC_CODE"),
                "csrc_desc": r.get("CSRC_CODE_DESC"),
                "area_name": r.get("AREA_NAME"),
                "area_desc": r.get("AREA_NAME_DESC"),
                "list_board": r.get("LIST_BOARD"),
                "list_date": _parse_date(r.get("LIST_DATE")),
                "delist_date": _parse_date(r.get("DELIST_DATE")),
                "state_code": r.get("STATE_CODE"),
                "product_status": r.get("PRODUCT_STATUS"),
                "market": "SSE",
            }
        )

    logger.debug("上交所解析完成: 有效 %d 条, 跳过 %d 条", len(out), skipped)
    return out


def fetch_from_announcements() -> list[dict]:
    """从巨潮公告数据反推公司列表（兜底方案）。

    深交所和北交所没有稳定的公开列表接口，用公告数据里的公司去重得到。
    """
    files = sorted(RAW_DIR.glob("ann_*.json"))
    if not files:
        logger.warning("没有找到公告数据，跳过反推")
        return []

    logger.info("从 %d 个公告文件反推公司列表 ...", len(files))

    seen: dict[str, dict] = {}
    for i, fp in enumerate(files, 1):
        try:
            records = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning("读取失败 %s: %s", fp.name, e)
            continue

        for r in records:
            code = r.get("sec_code")
            name = r.get("sec_name")
            if not code or not name or code in seen:
                continue
            seen[code] = {
                "sec_code": code,
                "sec_name": name,
                "company_abbr": name,
                "full_name": None,
                "full_name_en": None,
                "csrc_code": None,
                "csrc_desc": None,
                "area_name": None,
                "area_desc": None,
                "list_board": None,
                "list_date": None,
                "delist_date": None,
                "state_code": None,
                "product_status": None,
                "market": _infer_market(code),
            }

        if i % 20 == 0 or i == len(files):
            logger.debug("反推进度 %d/%d，已收集 %d 家", i, len(files), len(seen))

    logger.info("反推完成，共 %d 家", len(seen))
    return list(seen.values())


def _infer_market(code: str) -> str:
    """按代码前缀推断市场。"""
    if code.startswith("6"):
        return "SSE"
    if code.startswith(("8", "9", "4")):
        return "BJ"
    return "SZSE"


def _parse_date(s: str | None) -> str | None:
    """把 YYYYMMDD 转成 YYYY-MM-DD。"""
    if not s or s == "-":
        return None
    s = str(s).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return None


def main() -> None:
    log_banner(logger, "采集上市公司基本信息")

    merged: dict[str, dict] = {}

    # 1) 上交所（完整信息）
    try:
        sse_records = fetch_sse()
        for r in sse_records:
            merged[r["sec_code"]] = r
        logger.info("上交所导入 %d 家", len(sse_records))
    except Exception as e:  # noqa: BLE001
        logger.warning("上交所失败，将完全依赖公告反推: %s", e)

    # 2) 从公告数据补充（主要是深市和北交所）
    extra = fetch_from_announcements()
    added = 0
    for r in extra:
        if r["sec_code"] not in merged:
            merged[r["sec_code"]] = r
            added += 1
    logger.info("从公告数据补充 %d 家", added)

    # 3) 落盘
    records = sorted(merged.values(), key=lambda x: x["sec_code"])
    fp = RAW_DIR / "companies.json"
    fp.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    # 4) 统计
    by_market: dict[str, int] = {}
    for r in records:
        m = r.get("market") or "?"
        by_market[m] = by_market.get(m, 0) + 1

    logger.info("")
    log_summary(
        logger,
        "采集结果:",
        {
            "公司总数": f"{len(records):,} 家",
            **{f"  {m}": f"{c:,} 家" for m, c in sorted(by_market.items())},
            "输出文件": str(fp),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "公司信息采集异常终止")
        sys.exit(1)
