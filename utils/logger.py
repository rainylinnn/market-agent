"""统一日志系统。

设计目标：
1. **同时输出到控制台和文件** —— 控制台带颜色便于观察，文件保留完整记录
2. **按日期切分** —— 每天一个日志文件，便于追溯
3. **按级别分文件** —— ERROR 单独一份，便于快速定位问题
4. **结构化** —— 带时间戳、模块名、行号
5. **零配置** —— 任何模块 `from utils.logger import get_logger` 即可用

用法：
    from utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("开始处理")
    logger.debug("详细变量: %s", value)
    logger.warning("跳过 %d 条", n)
    logger.error("失败: %s", e, exc_info=True)

    # 任务上下文（自动附加到所有日志）
    with log_context(task="fetch_announcements", day="2026-10-30"):
        logger.info("处理中...")   # 日志里会带 [task=fetch_announcements day=2026-10-30]
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 级别 ----------
# 通过环境变量控制：LOG_LEVEL=DEBUG / INFO / WARNING / ERROR
DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------- 格式 ----------
# 控制台格式：更紧凑，带颜色
CONSOLE_FORMAT = "%(asctime)s %(levelname_short)s %(name_short)-18s %(message)s"
CONSOLE_DATE = "%H:%M:%S"

# 文件格式：完整信息，便于事后排查
FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
)
FILE_DATE = "%Y-%m-%d %H:%M:%S"

# ---------- 颜色 ----------
COLORS = {
    "DEBUG": "\033[36m",     # 青色
    "INFO": "\033[32m",      # 绿色
    "WARNING": "\033[33m",   # 黄色
    "ERROR": "\033[31m",     # 红色
    "CRITICAL": "\033[35m",  # 紫色
    "RESET": "\033[0m",
}

LEVEL_SHORT = {
    "DEBUG": "DBG",
    "INFO": "INF",
    "WARNING": "WRN",
    "ERROR": "ERR",
    "CRITICAL": "CRT",
}


class ColorFormatter(logging.Formatter):
    """控制台彩色格式化器。"""

    def __init__(self, fmt: str, datefmt: str, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        # 短级别名
        record.levelname_short = LEVEL_SHORT.get(record.levelname, record.levelname[:3])

        # 模块名简化：utils.db -> db
        name = record.name
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        record.name_short = name[:18]

        # 任务上下文
        ctx = getattr(record, "ctx", None)
        if ctx:
            record.msg = f"[{ctx}] {record.msg}"

        text = super().format(record)

        if self.use_color:
            color = COLORS.get(record.levelname, "")
            if color:
                # 只给级别上色
                text = text.replace(
                    record.levelname_short,
                    f"{color}{record.levelname_short}{COLORS['RESET']}",
                    1,
                )
        return text


class ContextFilter(logging.Filter):
    """把当前任务的上下文附加到日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _current_context.get()
        if ctx:
            record.ctx = " ".join(f"{k}={v}" for k, v in ctx.items())
        return True


# ---------- 任务上下文 ----------
import contextvars  # noqa: E402

_current_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)


@contextmanager
def log_context(**kwargs: Any):
    """在 with 块内给所有日志附加上下文。

    用法：
        with log_context(task="fetch", day="2026-10-30"):
            logger.info("...")   # 输出 [task=fetch day=2026-10-30] ...
    """
    token = _current_context.set({**_current_context.get(), **kwargs})
    try:
        yield
    finally:
        _current_context.reset(token)


# ---------- 初始化 ----------
_initialized = False


def _supports_color() -> bool:
    """判断终端是否支持颜色。"""
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


def setup_logging(level: str | None = None, force: bool = False) -> None:
    """初始化日志系统（幂等，重复调用无副作用）。"""
    global _initialized
    if _initialized and not force:
        return

    lvl = getattr(logging, (level or DEFAULT_LEVEL).upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handler 级别各自控制

    # 清掉已有 handler，避免重复
    for h in list(root.handlers):
        root.removeHandler(h)

    ctx_filter = ContextFilter()

    # --- 控制台 handler ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(lvl)
    console.setFormatter(
        ColorFormatter(CONSOLE_FORMAT, CONSOLE_DATE, use_color=_supports_color())
    )
    console.addFilter(ctx_filter)
    root.addHandler(console)

    # --- 全量文件 handler（按天切分，保留 30 天）---
    all_file = LOG_DIR / "app.log"
    fh_all = logging.handlers.TimedRotatingFileHandler(
        all_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    fh_all.suffix = "%Y-%m-%d"
    fh_all.setLevel(logging.DEBUG)
    fh_all.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE))
    fh_all.addFilter(ctx_filter)
    root.addHandler(fh_all)

    # --- 错误专用文件 ---
    err_file = LOG_DIR / "error.log"
    fh_err = logging.handlers.RotatingFileHandler(
        err_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    fh_err.setLevel(logging.ERROR)
    fh_err.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE))
    fh_err.addFilter(ctx_filter)
    root.addHandler(fh_err)

    # 降低第三方库噪音
    for noisy in ("urllib3", "requests", "pymilvus", "mysql", "pymysql"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _initialized = True

    logging.getLogger("utils.logger").debug(
        "日志系统初始化完成 level=%s dir=%s", logging.getLevelName(lvl), LOG_DIR
    )


def get_logger(name: str) -> logging.Logger:
    """获取 logger（自动初始化日志系统）。"""
    setup_logging()
    return logging.getLogger(name)


# ---------- 便捷函数：带分隔线的阶段标题 ----------
def log_banner(logger: logging.Logger, title: str, width: int = 72) -> None:
    """打印阶段标题（INFO 级别，写入日志文件）。"""
    logger.info("=" * width)
    logger.info("  %s", title)
    logger.info("=" * width)


def log_stage(logger: logging.Logger, index: int, total: int, title: str) -> None:
    """打印分步进度标题。"""
    logger.info("[%d/%d] %s", index, total, title)


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    prefix: str = "",
    every: int = 10,
) -> None:
    """按固定间隔打印进度，避免刷屏。"""
    if total <= 0:
        return
    if current % every == 0 or current == total:
        pct = current / total * 100
        logger.info("%s进度 %d/%d (%.1f%%)", prefix, current, total, pct)


def log_summary(logger: logging.Logger, title: str, items: dict[str, Any]) -> None:
    """打印汇总信息（键值对齐）。"""
    logger.info("%s", title)
    if not items:
        return
    width = max(len(str(k)) for k in items)
    for k, v in items.items():
        logger.info("  %-*s : %s", width, k, v)


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    context: str = "",
    *,
    reraise: bool = False,
) -> None:
    """统一异常记录格式。"""
    prefix = f"{context}: " if context else ""
    logger.error(
        "%s%s: %s",
        prefix,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    if reraise:
        raise exc


def get_log_stats() -> dict[str, Any]:
    """统计日志文件情况（便于自查）。"""
    stats: dict[str, Any] = {"dir": str(LOG_DIR), "files": []}
    if not LOG_DIR.exists():
        return stats
    for fp in sorted(LOG_DIR.glob("*.log*")):
        stats["files"].append(
            {
                "name": fp.name,
                "size_kb": round(fp.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(fp.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )
    return stats


if __name__ == "__main__":
    # 自测
    logger = get_logger("utils.logger")
    log_banner(logger, "日志系统自测")
    logger.debug("这是 debug 级别")
    logger.info("这是 info 级别")
    logger.warning("这是 warning 级别")
    logger.error("这是 error 级别")

    with log_context(task="demo", day="2026-10-30"):
        logger.info("带上下文的日志")

    try:
        raise ValueError("演示异常")
    except ValueError as e:
        log_exception(logger, e, "演示")

    log_summary(logger, "统计:", {"总条数": 133957, "耗时": "22.4 分钟"})
    print()
    print("日志文件:")
    for f in get_log_stats()["files"]:
        print(f"  {f['name']:<30} {f['size_kb']:>8} KB  {f['modified']}")
