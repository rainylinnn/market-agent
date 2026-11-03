"""一键启动脚本：检查环境 → 启动所有服务。

用法：
    python scripts/start_all.py            # 启动全部
    python scripts/start_all.py --check    # 只检查环境
    python scripts/start_all.py --stop     # 停止所有服务
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import EMBEDDING, LLM, MILVUS, MYSQL, ROOT
from utils.logger import get_logger, log_banner, log_exception, log_summary

logger = get_logger(__name__)

PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
LOG_DIR = ROOT / "logs"

AGENTS = [
    ("公告检索", "a2a_server.announcement_agent", 5005),
    ("公司画像", "a2a_server.company_agent", 5006),
    ("市场数据", "a2a_server.market_agent", 5007),
]


def check_port(port: int) -> bool:
    """检查端口是否已被占用（服务在运行）。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def check_environment() -> bool:
    """检查所有依赖服务。"""
    log_banner(logger, "环境检查")
    ok = True

    # 1. MySQL
    try:
        from utils.db import connect

        conn = connect()
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION() AS v")
            ver = cur.fetchone()["v"]
            cur.execute("SELECT COUNT(*) AS c FROM announcements")
            n = cur.fetchone()["c"]
        conn.close()
        logger.info("  [OK] MySQL %s  %s 条公告", ver, f"{n:,}")
    except Exception as e:  # noqa: BLE001
        logger.error("  [FAIL] MySQL: %s", e)
        logger.info("         启动: net start MySQL80")
        ok = False

    # 2. Milvus
    try:
        from utils.milvus_client import get_client, get_stats

        client = get_client()
        stats = get_stats(client)
        logger.info("  [OK] Milvus 集合 %s", stats.get("行数", "?"))
    except Exception as e:  # noqa: BLE001
        logger.error("  [FAIL] Milvus: %s", e)
        logger.info("         启动: cd docker && docker compose up -d")
        ok = False

    # 3. LLM
    try:
        if not LLM["api_key"]:
            raise RuntimeError("QIANWEN_API_KEY 未设置")
        logger.info("  [OK] LLM 模型 %s", LLM["model"])
    except Exception as e:  # noqa: BLE001
        logger.error("  [FAIL] LLM: %s", e)
        logger.info("         设置: $env:QIANWEN_API_KEY=[Environment]::GetEnvironmentVariable('QIANWEN_API_KEY','User')")
        ok = False

    # 4. Embedding
    try:
        if not EMBEDDING["api_key"]:
            raise RuntimeError("API Key 未设置")
        logger.info("  [OK] Embedding 模型 %s", EMBEDDING["model"])
    except Exception as e:  # noqa: BLE001
        logger.error("  [FAIL] Embedding: %s", e)
        ok = False

    return ok


def start_agents() -> None:
    """启动所有 agent 服务。"""
    log_banner(logger, "启动 A2A 服务")

    LOG_DIR.mkdir(exist_ok=True)
    started = []

    for title, module, port in AGENTS:
        if check_port(port):
            logger.info("  [跳过] %s (端口 %d 已占用)", title, port)
            continue

        log_file = LOG_DIR / f"agent_{port}.log"
        err_file = LOG_DIR / f"agent_{port}.err.log"

        try:
            subprocess.Popen(
                [PYTHON, "-m", module],
                cwd=str(ROOT),
                stdout=open(log_file, "w", encoding="utf-8"),
                stderr=open(err_file, "w", encoding="utf-8"),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            started.append((title, port))
            logger.info("  [启动] %s (端口 %d)", title, port)
            time.sleep(2)
        except Exception as e:  # noqa: BLE001
            log_exception(logger, e, f"启动 {title} 失败")

    # 等待就绪
    if started:
        logger.info("")
        logger.info("等待服务就绪 ...")
        for _ in range(20):
            time.sleep(1)
            ready = sum(1 for _, port in started if check_port(port))
            if ready == len(started):
                break

        logger.info("")
        for title, port in started:
            status = "就绪" if check_port(port) else "未就绪"
            logger.info("  %-10s 端口 %d  %s", title, port, status)
            logger.info("     服务卡片: http://127.0.0.1:%d/.well-known/agent-card.json", port)


def stop_agents() -> None:
    """停止所有 agent 服务（按端口找进程）。"""
    log_banner(logger, "停止 A2A 服务")

    import socket

    for title, module, port in AGENTS:
        if not check_port(port):
            logger.info("  [未运行] %s (端口 %d)", title, port)
            continue

        try:
            if sys.platform == "win32":
                # Windows: 用 netstat 找 PID 再 kill
                out = subprocess.check_output(
                    f'netstat -ano | findstr ":{port} "', shell=True, text=True
                )
                pids = set()
                for line in out.strip().split("\n"):
                    parts = line.split()
                    if parts and parts[-1].isdigit():
                        pids.add(parts[-1])
                for pid in pids:
                    subprocess.run(
                        f"taskkill /F /PID {pid}",
                        shell=True,
                        capture_output=True,
                    )
                logger.info("  [停止] %s (端口 %d)", title, port)
            else:
                logger.warning("  非 Windows 环境请手动停止端口 %d", port)
        except Exception as e:  # noqa: BLE001
            logger.error("  停止 %s 失败: %s", title, e)


def main() -> None:
    p = argparse.ArgumentParser(description="一键启动/停止服务")
    p.add_argument("--check", action="store_true", help="只检查环境")
    p.add_argument("--stop", action="store_true", help="停止所有服务")
    args = p.parse_args()

    if args.stop:
        stop_agents()
        return

    log_banner(logger, "MarketAgent 启动器")

    ok = check_environment()

    if args.check:
        logger.info("")
        logger.info("环境检查结果: %s", "全部通过" if ok else "存在问题")
        sys.exit(0 if ok else 1)

    if not ok:
        logger.error("")
        logger.error("环境检查未通过，请先解决上述问题")
        sys.exit(1)

    logger.info("")
    start_agents()

    logger.info("")
    log_banner(logger, "服务已就绪")
    logger.info("  查询示例:")
    logger.info('    python -m a2a_server.orchestrator "今天大盘怎么样"')
    logger.info('    python -m a2a_server.orchestrator "浦发银行最近有什么公告"')
    logger.info("")
    logger.info("  停止服务: python scripts/start_all.py --stop")


if __name__ == "__main__":
    main()
