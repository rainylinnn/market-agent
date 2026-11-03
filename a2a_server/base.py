"""A2A 公共层：AgentCard 构建、Executor 基类、服务启动。

基于 a2a-sdk 1.3.0（protobuf 风格 + FastAPI 路由）。

⚠️ 与旧版（0.x）的关键差异（全部实测踩过）：
    1. AgentCard 是 protobuf message，字段是 supported_interfaces（复数）
    2. 没有 A2AStarletteApplication，改用 create_*_routes + add_a2a_routes_to_fastapi
    3. JSON-RPC 方法名是 gRPC 风格 `SendMessage`，不是 `message/send`
    4. 请求字段是 camelCase（messageId），role 枚举是 `ROLE_USER`
    5. **必须带 HTTP 头 `A2A-Version: 1.0`**，否则服务端按 0.3 处理并拒绝
    6. DefaultRequestHandler 需要传 agent_card 参数
    7. **同步场景直接 enqueue 一个 Message 对象即可**，
       不需要 Task + TaskUpdater（用 TaskUpdater 会报
       "Agent should enqueue Task before TaskStatusUpdateEvent event"）
"""
from __future__ import annotations

import json
import sys
import uuid
from abc import abstractmethod
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Message,
    Part,
    Role,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger, log_exception

logger = get_logger(__name__)

# 服务端口分配
PORTS = {
    "announcement": 5005,   # 公告检索
    "company": 5006,        # 公司画像
    "market": 5007,         # 市场数据
    "orchestrator": 8085,   # 编排层
}


def make_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    skills: list[dict[str, Any]],
    version: str = "1.0.0",
    streaming: bool = True,
) -> AgentCard:
    """构建 AgentCard。"""
    skill_objs = []
    for s in skills:
        skill_objs.append(
            AgentSkill(
                id=s["id"],
                name=s["name"],
                description=s.get("description", ""),
                tags=s.get("tags", []),
                examples=s.get("examples", []),
            )
        )

    return AgentCard(
        name=name,
        description=description,
        version=version,
        provider=AgentProvider(organization="MarketAgent", url="https://localhost"),
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=False,
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/json"],
        skills=skill_objs,
    )


def safe_json(obj: Any, indent: int | None = 2) -> str:
    """JSON 序列化，自动处理 Decimal / date / datetime。

    MySQL 的 DECIMAL 字段返回 Decimal，DATE 返回 date，都不能直接序列化。
    """

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        if isinstance(o, set):
            return list(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", errors="replace")
        return str(o)

    return json.dumps(obj, ensure_ascii=False, indent=indent, default=_default)


def extract_text(context: RequestContext) -> str:
    """从请求上下文中提取用户输入的文本。"""
    msg = context.message
    if not msg:
        return ""

    parts = getattr(msg, "parts", None) or []
    texts = []
    for p in parts:
        text = getattr(p, "text", None)
        if text:
            texts.append(text)

    return " ".join(texts).strip()


class BaseExecutor(AgentExecutor):
    """A2A Executor 基类（同步响应模式）。

    子类只需实现 handle(query, context) -> dict，返回：
        {
            "answer": "给用户的自然语言回答",
            "data": {...},          # 结构化数据（可选）
            "sources": [...]        # 来源引用（可选）
        }

    框架行为：enqueue 一个 Message 后，框架直接作为响应返回。
    """

    agent_name: str = "base"

    def __init__(self) -> None:
        self.logger = get_logger(f"a2a.{self.agent_name}")

    @abstractmethod
    async def handle(self, query: str, context: RequestContext) -> dict[str, Any]:
        """处理业务逻辑。子类实现。"""
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """A2A 标准入口（同步模式）。"""
        query = extract_text(context)
        task_id = (context.task_id or "unknown")[:8]
        context_id = context.context_id or str(uuid.uuid4())

        self.logger.info(
            "收到请求 task=%s context=%s query=%r",
            task_id,
            context_id[:8],
            query[:80],
        )

        try:
            result = await self.handle(query, context)
            answer = result.get("answer", "")
            sources = result.get("sources") or []
            data = result.get("data")

            # 拼装完整回答
            blocks = [answer]

            if sources:
                src_lines = ["", "---", "参考来源："]
                for i, s in enumerate(sources[:10], 1):
                    src_lines.append(f"{i}. {s}")
                blocks.append("\n".join(src_lines))

            if data:
                blocks.append("\n【结构化数据】\n" + safe_json(data))

            full_text = "\n".join(blocks)

            # 同步场景：直接入队一个 Message
            reply = Message(
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                task_id=context.task_id or "",
                role=Role.ROLE_AGENT,
                parts=[Part(text=full_text)],
            )
            await event_queue.enqueue_event(reply)

            self.logger.info(
                "响应完成 task=%s 长度=%d 来源数=%d",
                task_id,
                len(full_text),
                len(sources),
            )

        except Exception as e:  # noqa: BLE001
            log_exception(self.logger, e, f"处理失败 task={task_id}")
            # 出错也返回一条 Message，让客户端能拿到错误信息
            err_msg = Message(
                message_id=str(uuid.uuid4()),
                context_id=context_id,
                task_id=context.task_id or "",
                role=Role.ROLE_AGENT,
                parts=[Part(text=f"处理失败：{type(e).__name__}: {e}")],
            )
            await event_queue.enqueue_event(err_msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """取消（同步场景无长任务，仅记录）。"""
        task_id = (context.task_id or "?")[:8]
        self.logger.warning("收到取消请求 task=%s（同步模式无需取消）", task_id)


def build_app(card: AgentCard, executor: AgentExecutor):
    """构建 FastAPI 应用（含 AgentCard 路由 + JSON-RPC 路由）。"""
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes.agent_card_routes import create_agent_card_routes
    from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
    from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
    from a2a.server.tasks import InMemoryTaskStore
    from fastapi import FastAPI

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    app = FastAPI(title=card.name, description=card.description)

    card_routes = create_agent_card_routes(card)
    jsonrpc_routes = create_jsonrpc_routes(handler, rpc_url="/")

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=card_routes,
        jsonrpc_routes=jsonrpc_routes,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": card.name}

    return app


def run_server(
    card: AgentCard,
    executor: AgentExecutor,
    port: int,
    host: str = "0.0.0.0",
):
    """启动 A2A 服务。"""
    import uvicorn

    app = build_app(card, executor)

    logger.info("=" * 68)
    logger.info("启动 A2A 服务: %s", card.name)
    logger.info("  端口      : %d", port)
    logger.info("  AgentCard : http://127.0.0.1:%d/.well-known/agent-card.json", port)
    logger.info("  健康检查  : http://127.0.0.1:%d/health", port)
    logger.info("  Skills    : %s", [s.id for s in card.skills])
    logger.info("=" * 68)

    uvicorn.run(app, host=host, port=port, log_level="info")
