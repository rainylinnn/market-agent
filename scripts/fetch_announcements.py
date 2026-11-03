"""巨潮资讯公告下载器。

设计要点：
1. **按天 + 按板块双维度切片**
   - pageSize 固定 30（实测上限）
   - 深翻页到第 150 页会**回绕到第 1 页**（实测发现），所以单日最多只能拿 100 页 = 3,000 条
   - 解决：按 plate 板块拆分（sz/sh/bj），每个板块单独翻页，规避 3,000 条上限

2. **断点续传**：每天一个 checkpoint 文件，中断后从断点继续

3. **分层标记**：按标题把公告分为 L1/L2/L3，后续按层处理

4. **限流保护**：请求间隔 0.25s，429 时指数退避

5. **增量落盘**：每处理完一天就写一次 JSON，避免全部堆在内存

日志：
- 每次请求的 URL 参数在 DEBUG 级别记录
- 每天的处理结果在 INFO 级别记录
- 网络失败在 WARNING，最终失败在 ERROR
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    CHECKPOINT_DIR,
    CNINFO_REFERER,
    CNINFO_URL,
    DATE_END,
    DATE_START,
    MAX_RETRIES,
    PAGE_SIZE,
    RAW_DIR,
    REQUEST_INTERVAL,
    USER_AGENT,
    classify_announcement,
)
from utils.logger import (
    get_logger,
    log_banner,
    log_exception,
    log_progress,
    log_summary,
)

logger = get_logger(__name__)

# 板块拆分：规避单日 3,000 条翻页上限
PLATES = ["sz", "sh", "bj"]

# 实测：第 150 页会回绕到第 1 页，安全上限 100 页
MAX_PAGES_PER_PLATE = 100


class CninfoCrawler:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Referer": CNINFO_REFERER,
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        self._last_request = 0.0
        self.request_count = 0
        self.error_count = 0
        logger.debug("CninfoCrawler 初始化，间隔=%.2fs", REQUEST_INTERVAL)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)
        self._last_request = time.monotonic()

    def query_page(self, day: str, page: int, plate: str = "") -> dict:
        """查询某天、某板块的第 page 页。day 格式 YYYY-MM-DD。"""
        payload = {
            "pageNum": page,
            "pageSize": PAGE_SIZE,
            "column": "szse",
            "tabName": "fulltext",
            "plate": plate,
            "stock": "",
            "searchkey": "",
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": f"{day}~{day}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }

        logger.debug("请求 day=%s plate=%s page=%d", day, plate or "(全部)", page)

        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()
            try:
                resp = self.session.post(CNINFO_URL, data=payload, timeout=30)
                self.request_count += 1

                if resp.status_code == 429:
                    wait = 5 * attempt
                    logger.warning(
                        "限流 429，等待 %ds 后重试 (attempt=%d/%d)",
                        wait,
                        attempt,
                        MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                logger.debug(
                    "响应 day=%s page=%d 条数=%d",
                    day,
                    page,
                    len(data.get("announcements") or []),
                )
                return data

            except Exception as e:  # noqa: BLE001
                last_err = e
                self.error_count += 1
                if attempt < MAX_RETRIES:
                    backoff = 2**attempt
                    logger.warning(
                        "请求失败 (attempt=%d/%d)，%ds 后重试: %s",
                        attempt,
                        MAX_RETRIES,
                        backoff,
                        e,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "请求最终失败 day=%s plate=%s page=%d: %s",
                        day,
                        plate,
                        page,
                        e,
                    )

        raise RuntimeError(f"查询失败 {day} plate={plate} page={page}: {last_err}")

    def fetch_plate(self, day: str, plate: str, seen: set[str]) -> list[dict]:
        """拉取某天某板块的全部公告，去重写入 seen。"""
        out: list[dict] = []
        page = 1
        total = None
        prev_first_id = None

        while page <= MAX_PAGES_PER_PLATE:
            data = self.query_page(day, page, plate)

            if total is None:
                total = data.get("totalAnnouncement") or 0
                logger.debug("plate=%s total=%d", plate, total)
                if total == 0:
                    break

            anns = data.get("announcements") or []
            if not anns:
                logger.debug("plate=%s page=%d 空页，停止", plate, page)
                break

            # 回绕检测：如果本页首条 ID 与上一页首条相同，说明已经翻回去了
            first_id = anns[0].get("announcementId")
            if page > 1 and first_id == prev_first_id:
                logger.warning(
                    "检测到分页回绕 plate=%s page=%d，提前停止（这是已知的巨潮接口限制）",
                    plate,
                    page,
                )
                break
            prev_first_id = first_id

            for a in anns:
                aid = a.get("announcementId")
                if not aid or aid in seen:
                    continue
                seen.add(aid)

                title = (a.get("announcementTitle") or "").replace("<em>", "").replace("</em>", "")

                out.append(
                    {
                        "announcement_id": aid,
                        "sec_code": a.get("secCode"),
                        "sec_name": a.get("secName"),
                        "title": title,
                        "announcement_time": a.get("announcementTime"),
                        "pub_date": day,
                        "plate": plate,
                        "adjunct_url": a.get("adjunctUrl"),
                        "adjunct_size_kb": a.get("adjunctSize"),
                        "adjunct_type": a.get("adjunctType"),
                        "announcement_type": a.get("announcementTypeName"),
                        "level": classify_announcement(title),
                    }
                )

            if len(anns) < PAGE_SIZE:
                break
            if total and len(out) >= total:
                break

            page += 1

        logger.debug("plate=%s 完成 采集=%d 页数=%d", plate, len(out), page)
        return out

    def fetch_day(self, day: str) -> list[dict]:
        """按板块拆分拉取一整天的公告，合并去重。"""
        seen: set[str] = set()
        records: list[dict] = []
        for plate in PLATES:
            try:
                part = self.fetch_plate(day, plate, seen)
                records.extend(part)
                logger.debug("day=%s plate=%s 新增=%d", day, plate, len(part))
            except Exception as e:  # noqa: BLE001
                logger.error("day=%s plate=%s 采集失败: %s", day, plate, e)
        return records


def date_range(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        yield cur.isoformat()
        cur += timedelta(days=1)


def load_checkpoint(day: str) -> bool:
    return (CHECKPOINT_DIR / f"day_{day}.done").exists()


def mark_done(day: str, count: int) -> None:
    (CHECKPOINT_DIR / f"day_{day}.done").write_text(
        json.dumps({"day": day, "count": count, "at": datetime.now().isoformat()}),
        encoding="utf-8",
    )
    logger.debug("断点已标记 day=%s count=%d", day, count)


def run(start: str = DATE_START, end: str = DATE_END, force: bool = False) -> None:
    crawler = CninfoCrawler()
    days = list(date_range(start, end))
    total_days = len(days)

    log_banner(logger, f"巨潮资讯公告下载  {start} ~ {end}  共 {total_days} 天")
    logger.info("板块拆分: %s（规避单日 3000 条上限）", ", ".join(PLATES))
    logger.info("分页大小: %d 条/页   请求间隔: %.2fs   重试次数: %d", PAGE_SIZE, REQUEST_INTERVAL, MAX_RETRIES)

    all_stats = {"L1": 0, "L2": 0, "L3": 0}
    grand_total = 0
    t_start = time.time()
    skipped = 0
    failed_days: list[str] = []

    for i, day in enumerate(days, 1):
        if not force and load_checkpoint(day):
            skipped += 1
            fp = RAW_DIR / f"ann_{day}.json"
            if fp.exists():
                try:
                    prev = json.loads(fp.read_text(encoding="utf-8"))
                    for r in prev:
                        all_stats[r["level"]] = all_stats.get(r["level"], 0) + 1
                    grand_total += len(prev)
                except Exception as e:  # noqa: BLE001
                    logger.warning("读取已完成文件失败 %s: %s", fp.name, e)
            logger.debug("[%d/%d] %s 已完成，跳过", i, total_days, day)
            continue

        t0 = time.time()
        try:
            records = crawler.fetch_day(day)
        except Exception as e:  # noqa: BLE001
            log_exception(logger, e, f"[{i}/{total_days}] {day} 采集失败")
            failed_days.append(day)
            continue

        fp = RAW_DIR / f"ann_{day}.json"
        fp.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
        mark_done(day, len(records))

        l1 = sum(1 for r in records if r["level"] == "L1")
        for r in records:
            all_stats[r["level"]] = all_stats.get(r["level"], 0) + 1
        grand_total += len(records)

        dt = time.time() - t0
        elapsed = time.time() - t_start
        done = i - skipped
        eta = (elapsed / done) * (total_days - i) if done else 0

        logger.info(
            "[%3d/%d] %s  %5d 条  L1=%3d  %5.1fs  剩余约 %.0f 分钟",
            i,
            total_days,
            day,
            len(records),
            l1,
            dt,
            eta / 60,
        )

    # ---------- 汇总 ----------
    elapsed_min = (time.time() - t_start) / 60
    logger.info("")
    log_banner(logger, "采集完成")
    log_summary(
        logger,
        "统计:",
        {
            "公告总数": f"{grand_total:,} 条",
            "L1 核心": f"{all_stats.get('L1', 0):,} 条",
            "L2 重要": f"{all_stats.get('L2', 0):,} 条",
            "L3 模板": f"{all_stats.get('L3', 0):,} 条",
            "跳过天数": f"{skipped} 天（已完成）",
            "失败天数": f"{len(failed_days)} 天" + (f" {failed_days}" if failed_days else ""),
            "HTTP 请求": f"{crawler.request_count:,} 次",
            "请求错误": f"{crawler.error_count:,} 次",
            "总耗时": f"{elapsed_min:.1f} 分钟",
            "平均速度": f"{grand_total/max(elapsed_min,0.01):,.0f} 条/分钟",
        },
    )

    summary_fp = RAW_DIR / "_summary.json"
    summary_fp.write_text(
        json.dumps(
            {
                "date_start": start,
                "date_end": end,
                "total": grand_total,
                "by_level": all_stats,
                "failed_days": failed_days,
                "http_requests": crawler.request_count,
                "http_errors": crawler.error_count,
                "elapsed_minutes": round(elapsed_min, 2),
                "finished_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("摘要已存: %s", summary_fp)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="巨潮资讯公告下载")
    p.add_argument("--start", default=DATE_START, help="开始日期 YYYY-MM-DD")
    p.add_argument("--end", default=DATE_END, help="结束日期 YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="忽略断点，重新下载")
    args = p.parse_args()

    try:
        run(args.start, args.end, args.force)
    except KeyboardInterrupt:
        logger.warning("用户中断（Ctrl+C），已完成的天数已保存，下次运行会跳过")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "采集任务异常终止")
        sys.exit(1)
