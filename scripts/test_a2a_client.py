"""A2A 客户端测试：直接调用各 agent。

用法：
    python scripts/test_a2a_client.py                    # 测试全部
    python scripts/test_a2a_client.py --agent market     # 只测一个
    python scripts/test_a2a_client.py --query "黄金价格"  # 自定义查询
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger, log_banner, log_exception

logger = get_logger(__name__)

AGENT_PORTS = {
    "announcement": 5005,
    "company": 5006,
    "market": 5007,
}

# 每个 agent 的测试查询
TEST_QUERIES = {
    "announcement": ["最近有哪些公司回购股份", "查找重大资产重组公告"],
    "company": ["介绍一下浦发银行", "600000 是什么公司"],
    "market": ["今天大盘怎么样", "人民币兑美元汇率", "黄金价格"],
}


def call_agent(
    port: int,
    query: str,
    timeout: int = 120,
) -> dict:
    """通过 JSON-RPC 调用 agent。

    ⚠️ A2A v1.0 与 v0.3 的关键差异（实测踩坑）：
        - 方法名：`SendMessage`（gRPC 风格），不是 `message/send`
        - 字段名：camelCase（messageId），不是 snake_case（message_id）
        - role 枚举：`ROLE_USER`，不是 `user`
    """
    url = f"http://127.0.0.1:{port}/"

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "ROLE_USER",
                "parts": [{"text": query}],
            }
        },
    }

    t0 = time.time()
    # 必须带版本头，否则服务端按 0.3 处理并拒绝
    resp = requests.post(
        url,
        json=payload,
        headers={"A2A-Version": "1.0", "Content-Type": "application/json"},
        timeout=timeout,
    )
    elapsed = (time.time() - t0) * 1000

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"JSON-RPC 错误: {data['error']}")

    return {"data": data.get("result", {}), "elapsed_ms": elapsed}


def extract_answer(result: dict) -> str:
    """从 A2A 响应中提取文本答案。

    A2A v1.0 同步模式下，响应是 `{"message": {...}}`（单条 Message），
    不是 Task + artifacts。这里兼容多种形态。
    """
    texts = []

    # 形态 1：同步 Message 响应
    msg = result.get("message")
    if msg:
        for part in msg.get("parts") or []:
            if part.get("text"):
                texts.append(part["text"])
        if texts:
            return "\n".join(texts)

    # 形态 2：Task + artifacts（长任务）
    artifacts = result.get("artifacts") or []
    for art in artifacts:
        for part in art.get("parts") or []:
            if part.get("text"):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts)

    # 形态 3：Task 状态消息
    status = result.get("status") or {}
    st_msg = status.get("message") or {}
    for part in st_msg.get("parts") or []:
        if part.get("text"):
            texts.append(part["text"])
    if texts:
        return "\n".join(texts)

    # 形态 4：history 里最后一条 agent 消息
    for h in reversed(result.get("history") or []):
        if h.get("role") in ("ROLE_AGENT", "agent"):
            for part in h.get("parts") or []:
                if part.get("text"):
                    texts.append(part["text"])
            if texts:
                break

    return "\n".join(texts)


def test_agent(name: str, queries: list[str]) -> bool:
    """测试单个 agent。"""
    port = AGENT_PORTS[name]
    logger.info("")
    logger.info("=" * 70)
    logger.info("  测试 Agent: %s (端口 %d)", name, port)
    logger.info("=" * 70)

    # 先检查 Agent Card
    try:
        r = requests.get(
            f"http://127.0.0.1:{port}/.well-known/agent-card.json", timeout=10
        )
        card = r.json()
        logger.info("  Agent Card: %s", card.get("name"))
        logger.info("  Skills: %s", [s.get("id") for s in card.get("skills", [])])
    except Exception as e:  # noqa: BLE001
        logger.error("  无法获取 Agent Card: %s", e)
        return False

    all_ok = True
    for q in queries:
        logger.info("")
        logger.info("  查询: %s", q)
        try:
            r = call_agent(port, q)
            answer = extract_answer(r["data"])

            if not answer:
                logger.warning("    [!] 无回答内容，原始响应:")
                logger.warning("    %s", json.dumps(r["data"], ensure_ascii=False)[:400])
                all_ok = False
                continue

            # 判断是否是业务失败（Executor 内部异常）
            if answer.startswith("处理失败"):
                logger.error("    业务失败: %s", answer[:200])
                all_ok = False
                continue

            # 打印前 600 字
            preview = answer[:600].replace("\n", "\n    ")
            logger.info("    回答（%.0fms）:", r["elapsed_ms"])
            logger.info("    %s", preview)
            if len(answer) > 600:
                logger.info("    ... (共 %d 字)", len(answer))

        except Exception as e:  # noqa: BLE001
            log_exception(logger, e, "    调用失败")
            all_ok = False

    return all_ok


def main() -> None:
    p = argparse.ArgumentParser(description="A2A 客户端测试")
    p.add_argument("--agent", help="只测指定 agent")
    p.add_argument("--query", help="自定义查询（需配合 --agent）")
    args = p.parse_args()

    log_banner(logger, "A2A 客户端测试")

    if args.agent:
        if args.agent not in AGENT_PORTS:
            logger.error("未知 agent: %s", args.agent)
            sys.exit(1)
        queries = [args.query] if args.query else TEST_QUERIES[args.agent]
        ok = test_agent(args.agent, queries)
    else:
        results = {}
        for name in AGENT_PORTS:
            results[name] = test_agent(name, TEST_QUERIES[name])
        ok = all(results.values())

    logger.info("")
    log_banner(logger, "测试结果")
    if args.agent:
        logger.info("  %s: %s", args.agent, "通过" if ok else "失败")
    else:
        for name, passed in results.items():
            logger.info("  %-14s %s", name, "通过" if passed else "失败")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
