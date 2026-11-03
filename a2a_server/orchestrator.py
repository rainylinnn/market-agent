"""编排层：意图识别 + 槽位填充 + 任务规划 + A2A 调度 + 结果汇总。

流程：
    用户 query
       ↓
    1. 意图识别（LLM）：识别出要调用哪些 agent
       ↓
    2. 槽位检查：必要信息是否齐全，缺失则追问
       ↓
    3. 任务规划：决定调用顺序（并行/串行）
       ↓
    4. A2A 调度：并发调用各 agent（每个 agent 是独立进程）
       ↓
    5. 结果汇总（LLM）：把多个 agent 的结果整合成一个回答

与 smartVoyage 的差异：
    - 路由不是硬编码的，而是从 Agent Card 动态生成
    - agent 是真正的独立进程，通过 A2A 协议通信
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a_server.base import PORTS
from utils.config import LLM
from utils.llm import chat_json
from utils.logger import get_logger, log_banner, log_exception, log_summary

logger = get_logger(__name__)

# agent 端口映射
AGENT_PORTS = {
    "announcement": PORTS["announcement"],
    "company": PORTS["company"],
    "market": PORTS["market"],
}

AGENT_DESC = {
    "announcement": "公告检索：查找回购、重组、减持、业绩预告等各类公告",
    "company": "公司画像：查询公司基本信息、行业、地区、公告统计",
    "market": "市场数据：查询指数行情、人民币汇率、黄金价格",
}


class Orchestrator:
    """多智能体编排器。"""

    def __init__(self) -> None:
        self.logger = get_logger("orchestrator")
        self._cards: dict[str, dict] = {}

    # ---------- Agent Card 动态发现 ----------

    def discover_agents(self) -> dict[str, dict]:
        """从各 agent 拉取 Agent Card（动态发现能力）。"""
        cards = {}
        for name, port in AGENT_PORTS.items():
            try:
                r = requests.get(
                    f"http://127.0.0.1:{port}/.well-known/agent-card.json", timeout=5
                )
                if r.status_code == 200:
                    cards[name] = r.json()
                    self.logger.debug("发现 agent: %s (%d skills)", name, len(cards[name].get("skills", [])))
                else:
                    self.logger.warning("agent %s 返回 HTTP %d", name, r.status_code)
            except Exception as e:  # noqa: BLE001
                self.logger.warning("agent %s 不可达: %s", name, e)

        self._cards = cards
        return cards

    def build_routing_prompt(self) -> str:
        """用 Agent Card 动态生成路由提示。"""
        lines = []
        for name, card in self._cards.items():
            skills = card.get("skills") or []
            skill_desc = "；".join(
                f"{s.get('name')}（{s.get('description')}）" for s in skills[:3]
            )
            lines.append(f"- {name}: {card.get('name')} —— {skill_desc}")
        return "\n".join(lines) if lines else "（无可用 agent）"

    # ---------- 意图识别 ----------

    def recognize_intent(self, query: str) -> dict[str, Any]:
        """识别意图和槽位。"""
        routing = self.build_routing_prompt()

        prompt = f"""你是投研助手的意图识别模块。根据用户问题，判断需要调用哪些 agent。

## 可用 agent
{routing}

## 用户问题
{query}

## 输出要求
严格输出 JSON，格式：
{{
  "intents": ["announcement", "company", "market"],   // 要调用的 agent，按重要性排序
  "reason": "为什么选这些",
  "missing": [],                                      // 缺失的必要信息（如公司名）
  "follow_up": ""                                      // 如果信息不足，给用户的追问
}}

规则：
- intents 只能从上面列出的 agent 名称里选
- 如果问题与投研无关，intents 返回空数组
- 如果缺公司名等必要信息无法检索，missing 里说明，follow_up 给追问话术

## 多选原则（重要，宁少勿多）
只有当用户一句话里存在**两个及以上彼此独立的问题**时，才选多个 agent。
判断方法：能不能把这句话拆成两个各自独立、单独问也成立的句子？

选多个的例子：
  "浦发银行最近有什么公告，股价怎么样"
    → 拆成「浦发银行最近有什么公告」+「浦发银行股价怎么样」→ announcement + market
  "今天大盘怎么样，浦发银行是什么公司"
    → 拆成两个独立问题 → market + company

只选一个的例子（这类问题看起来涉及多个领域，但实质是一个问题）：
  "贵州燃气发了多少公告"
    → 只有一个诉求：数公告。公告数量由 company 的统计能力直接给出，
      不需要再调 announcement 去检索 → 只选 company
  "688728 最近有什么公告"
    → 只有一个诉求：查这家公司的公告。公司名解析在 announcement 内部完成
      （它自己会做代码/名称识别），不需要 company 配合 → 只选 announcement
  "浦发银行是什么公司"
    → 只选 company

核心判断：某个 agent 能独立完成整个诉求时，就只选它，不要为了"更全面"而叠加。
"""

        try:
            result = chat_json(prompt, default=None)
            if not result:
                return self._fallback_intent(query)

            intents = [i for i in result.get("intents", []) if i in AGENT_PORTS]
            return {
                "intents": intents,
                "reason": result.get("reason", ""),
                "missing": result.get("missing", []),
                "follow_up": result.get("follow_up", ""),
                "source": "llm",
            }
        except Exception as e:  # noqa: BLE001
            log_exception(self.logger, e, "意图识别失败，使用规则兜底")
            return self._fallback_intent(query)

    def _fallback_intent(self, query: str) -> dict[str, Any]:
        """规则兜底（LLM 不可用时）。"""
        intents = []
        if any(k in query for k in ["公告", "回购", "重组", "减持", "增持", "业绩", "分红", "停牌"]):
            intents.append("announcement")
        if any(k in query for k in ["公司", "介绍", "行业", "上市", "是什么"]):
            intents.append("company")
        if any(k in query for k in ["行情", "指数", "大盘", "汇率", "黄金", "股价"]):
            intents.append("market")

        if not intents:
            intents = ["announcement"]

        return {"intents": intents, "reason": "规则匹配", "missing": [], "follow_up": "", "source": "rule"}

    # ---------- A2A 调用 ----------

    def call_agent(self, name: str, query: str, timeout: int = 120) -> dict[str, Any]:
        """调用单个 agent（A2A 协议）。"""
        port = AGENT_PORTS[name]
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

        try:
            r = requests.post(
                url,
                json=payload,
                headers={"A2A-Version": "1.0", "Content-Type": "application/json"},
                timeout=timeout,
            )
            r.raise_for_status()
            data = r.json()

            if "error" in data:
                return {"ok": False, "error": str(data["error"]), "agent": name}

            result = data.get("result", {})
            answer = self._extract_text(result)

            if answer.startswith("处理失败"):
                return {"ok": False, "error": answer, "agent": name}

            return {"ok": True, "agent": name, "answer": answer, "raw": result}

        except Exception as e:  # noqa: BLE001
            self.logger.error("调用 %s 失败: %s", name, e)
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "agent": name}

    @staticmethod
    def _extract_text(result: dict) -> str:
        """从 A2A 响应提取文本。"""
        msg = result.get("message")
        if msg:
            parts = [p.get("text", "") for p in (msg.get("parts") or [])]
            return "\n".join(t for t in parts if t)

        for art in result.get("artifacts") or []:
            parts = [p.get("text", "") for p in (art.get("parts") or [])]
            text = "\n".join(t for t in parts if t)
            if text:
                return text

        return ""

    async def dispatch_parallel(self, intents: list[str], query: str) -> dict[str, dict]:
        """并发调用多个 agent。"""
        if not intents:
            return {}

        self.logger.info("并发调用 %d 个 agent: %s", len(intents), intents)

        tasks = [
            asyncio.to_thread(self.call_agent, name, query) for name in intents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = {}
        for name, res in zip(intents, results):
            if isinstance(res, Exception):
                out[name] = {"ok": False, "error": str(res), "agent": name}
            else:
                out[name] = res
            status = "OK" if out[name].get("ok") else "FAIL"
            self.logger.debug("  %s: %s", name, status)

        return out

    # ---------- 结果汇总 ----------

    def synthesize(
        self, query: str, results: dict[str, dict], intents: list[str]
    ) -> str:
        """把多 agent 结果汇总成一个回答。

        只有一个 agent 成功时，直接返回它的答案（避免 LLM 二次加工引入幻觉）。
        多个 agent 成功时，用 LLM 整合。
        """
        ok_results = {k: v for k, v in results.items() if v.get("ok")}

        if not ok_results:
            errs = "; ".join(f"{k}: {v.get('error', '?')[:60]}" for k, v in results.items())
            return f"抱歉，所有查询都失败了。\n错误信息：{errs}"

        # 单个成功：直接返回
        if len(ok_results) == 1:
            name, res = next(iter(ok_results.items()))
            self.logger.debug("单 agent 结果直接返回: %s", name)
            return res["answer"]

        # 多个成功：LLM 汇总
        self.logger.info("汇总 %d 个 agent 的结果", len(ok_results))

        sections = []
        for name, res in ok_results.items():
            # 去掉结构化数据部分，只保留自然语言回答
            text = res["answer"].split("【结构化数据】")[0].strip()
            sections.append(f"### {AGENT_DESC.get(name, name)}\n{text}")

        joined = "\n\n".join(sections)

        prompt = f"""你是投研分析师。下面是多个专业助手对同一个问题的回答，请整合成一份连贯的分析。

## 用户问题
{query}

## 各助手回答
{joined}

## 要求
1. 直接给出整合后的答案，不要重复"根据XX助手"这类表述
2. 保留所有关键数字和事实
3. 如果不同来源有矛盾，明确指出
4. 结构清晰，用分段或列表
5. 不要编造数据

直接输出整合后的回答：
"""

        try:
            from utils.llm import chat

            return chat(prompt, temperature=0.2, max_tokens=1500)
        except Exception as e:  # noqa: BLE001
            log_exception(self.logger, e, "LLM 汇总失败，返回原始拼接")
            return joined

    # ---------- 主流程 ----------

    async def arun(self, query: str) -> dict[str, Any]:
        """完整流程（异步版本）。

        供 FastAPI 等已在事件循环中的环境使用。
        CLI 场景请用 run()。
        """
        self.logger.info("=" * 60)
        self.logger.info("用户问题: %s", query)

        # 1. 动态发现 agent（同步 HTTP，放线程池）
        cards = await asyncio.to_thread(self.discover_agents)
        if not cards:
            return {
                "answer": "所有 agent 服务都不可用，请先启动：python scripts/run_agents.py",
                "intents": [],
                "results": {},
            }
        self.logger.info("发现 %d 个可用 agent: %s", len(cards), list(cards))

        # 2. 意图识别（LLM 调用是同步的，放线程池）
        intent_result = await asyncio.to_thread(self.recognize_intent, query)
        intents = intent_result["intents"]
        self.logger.info(
            "意图: %s  理由: %s  来源: %s",
            intents,
            intent_result.get("reason", "")[:60],
            intent_result.get("source"),
        )

        # 3. 信息不足则追问
        if intent_result.get("follow_up"):
            self.logger.info("信息不足，需要追问")
            return {
                "answer": intent_result["follow_up"],
                "intents": intents,
                "results": {},
                "need_follow_up": True,
            }

        if not intents:
            return {
                "answer": "这个问题看起来和 A 股投研无关。我可以帮你查询：\n"
                "· 上市公司公告（回购、重组、减持等）\n"
                "· 公司基本信息与公告统计\n"
                "· 市场行情、汇率、黄金",
                "intents": [],
                "results": {},
            }

        # 4. 并发调度（直接 await，不再 asyncio.run）
        results = await self.dispatch_parallel(intents, query)

        ok_count = sum(1 for r in results.values() if r.get("ok"))
        self.logger.info("调度完成: %d/%d 成功", ok_count, len(intents))

        # 5. 汇总（LLM 调用同步，放线程池）
        answer = await asyncio.to_thread(self.synthesize, query, results, intents)

        return {
            "answer": answer,
            "intents": intents,
            "reason": intent_result.get("reason"),
            "results": {
                k: {
                    "ok": v.get("ok"),
                    "chars": len(v.get("answer", "")) if v.get("ok") else 0,
                    "error": v.get("error") if not v.get("ok") else None,
                }
                for k, v in results.items()
            },
        }

    def run(self, query: str) -> dict[str, Any]:
        """完整流程（同步版本，供 CLI 使用）。

        内部用 asyncio.run 驱动 arun()。
        注意：不能在已有事件循环中调用（FastAPI 场景请直接用 arun）。
        """
        return asyncio.run(self.arun(query))

    def _run_sync_legacy(self, query: str) -> dict[str, Any]:
        """旧版同步实现（保留供参考，不再使用）。"""
        self.logger.info("=" * 60)
        self.logger.info("用户问题: %s", query)

        # 1. 动态发现 agent
        cards = self.discover_agents()
        if not cards:
            return {
                "answer": "所有 agent 服务都不可用，请先启动：python scripts/run_agents.py",
                "intents": [],
                "results": {},
            }
        self.logger.info("发现 %d 个可用 agent: %s", len(cards), list(cards))

        # 2. 意图识别
        intent_result = self.recognize_intent(query)
        intents = intent_result["intents"]
        self.logger.info(
            "意图: %s  理由: %s  来源: %s",
            intents,
            intent_result.get("reason", "")[:60],
            intent_result.get("source"),
        )

        # 3. 信息不足则追问
        if intent_result.get("follow_up"):
            self.logger.info("信息不足，需要追问")
            return {
                "answer": intent_result["follow_up"],
                "intents": intents,
                "results": {},
                "need_follow_up": True,
            }

        if not intents:
            return {
                "answer": "这个问题看起来和 A 股投研无关。我可以帮你查询：\n"
                "· 上市公司公告（回购、重组、减持等）\n"
                "· 公司基本信息与公告统计\n"
                "· 市场行情、汇率、黄金",
                "intents": [],
                "results": {},
            }

        # 4. 并发调度
        results = asyncio.run(self.dispatch_parallel(intents, query))

        ok_count = sum(1 for r in results.values() if r.get("ok"))
        self.logger.info("调度完成: %d/%d 成功", ok_count, len(intents))

        # 5. 汇总
        answer = self.synthesize(query, results, intents)

        return {
            "answer": answer,
            "intents": intents,
            "reason": intent_result.get("reason"),
            "results": {
                k: {
                    "ok": v.get("ok"),
                    "chars": len(v.get("answer", "")) if v.get("ok") else 0,
                    "error": v.get("error") if not v.get("ok") else None,
                }
                for k, v in results.items()
            },
        }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="多智能体编排器")
    p.add_argument("query", nargs="?", help="要查询的问题")
    p.add_argument("--discover", action="store_true", help="只测试 agent 发现")
    args = p.parse_args()

    orch = Orchestrator()

    if args.discover:
        log_banner(logger, "Agent 动态发现")
        cards = orch.discover_agents()
        for name, card in cards.items():
            logger.info("  %-14s %s", name, card.get("name"))
            for s in card.get("skills", []):
                logger.info("      · %s: %s", s.get("id"), s.get("name"))
        if not cards:
            logger.error("没有发现任何 agent，请先启动服务")
        return

    if not args.query:
        logger.error("请提供查询内容，例如：python -m a2a_server.orchestrator \"浦发银行最近有什么公告\"")
        sys.exit(1)

    log_banner(logger, "多智能体编排")
    t0 = __import__("time").time()

    result = orch.run(args.query)

    elapsed = __import__("time").time() - t0

    print()
    print("=" * 72)
    print(result["answer"])
    print("=" * 72)
    print()

    log_summary(
        logger,
        "执行信息:",
        {
            "意图": result["intents"],
            "耗时": f"{elapsed:.1f}s",
            "各 agent": result["results"],
        },
    )


if __name__ == "__main__":
    main()
