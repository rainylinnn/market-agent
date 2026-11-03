"""启动所有 A2A agent 服务。

用法：
    python scripts/run_agents.py              # 启动全部（前台，Ctrl+C 停止）
    python scripts/run_agents.py --only market # 只启动一个
    python scripts/run_agents.py --list        # 列出可用服务

端口分配：
    5005  公告检索
    5006  公司画像
    5007  市场数据
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger, log_banner, log_exception

logger = get_logger(__name__)

AGENTS = {
    "announcement": ("a2a_server.announcement_agent", 5005, "公告检索"),
    "company": ("a2a_server.company_agent", 5006, "公司画像"),
    "market": ("a2a_server.market_agent", 5007, "市场数据"),
}


def run_one(module_name: str, port: int, title: str) -> None:
    """在子进程中启动一个 agent。"""
    import importlib

    try:
        mod = importlib.import_module(module_name)
        card = mod.create_card(port)
        executor_cls = None
        for name in dir(mod):
            if name.endswith("Executor") and name != "BaseExecutor":
                executor_cls = getattr(mod, name)
                break

        if not executor_cls:
            logger.error("%s 找不到 Executor 类", module_name)
            return

        from a2a_server.base import run_server

        run_server(card, executor_cls(), port=port)

    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, f"{title} 启动失败")


def main() -> None:
    p = argparse.ArgumentParser(description="启动 A2A agent 服务")
    p.add_argument("--only", help="只启动指定服务（announcement/company/market）")
    p.add_argument("--list", action="store_true", help="列出可用服务")
    args = p.parse_args()

    if args.list:
        logger.info("可用服务:")
        for key, (mod, port, title) in AGENTS.items():
            logger.info("  %-14s %s  端口=%d  %s", key, mod, port, title)
        return

    targets = {args.only: AGENTS[args.only]} if args.only else AGENTS

    if args.only and args.only not in AGENTS:
        logger.error("未知服务: %s（可选: %s）", args.only, ", ".join(AGENTS.keys()))
        sys.exit(1)

    log_banner(logger, f"启动 {len(targets)} 个 A2A 服务")

    procs: list[mp.Process] = []
    for key, (mod, port, title) in targets.items():
        logger.info("  %-14s 端口=%d  %s", key, port, title)
        proc = mp.Process(target=run_one, args=(mod, port, title), daemon=False)
        proc.start()
        procs.append(proc)
        time.sleep(1.5)  # 错开启动，避免日志混乱

    logger.info("")
    logger.info("所有服务已启动，按 Ctrl+C 停止")
    logger.info("  Agent Card 检查:")
    for key, (mod, port, title) in targets.items():
        logger.info("    http://127.0.0.1:%d/.well-known/agent-card.json", port)

    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        logger.warning("收到中断，正在停止所有服务 ...")
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(timeout=5)
        logger.info("已停止")


if __name__ == "__main__":
    main()
