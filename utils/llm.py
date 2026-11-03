"""千问 LLM 客户端封装。

通过 OpenAI 兼容接口调用，支持：
- 自动重试（指数退避）
- 完整日志（请求、响应、token 用量）
- 结构化 JSON 输出（自动清洗 markdown 代码块）
- 流式输出（可选）
- 用量统计

环境变量：
    QIANWEN_API_KEY   API Key（必需）
    QIANWEN_BASE_URL  接口地址（默认阿里云百炼北京）
    QIANWEN_MODEL     模型名（默认 qwen3.8-27b）

用法：
    from utils.llm import chat, chat_json

    # 普通对话
    answer = chat("用一句话解释什么是回购")

    # 结构化输出
    data = chat_json("把这句话拆成意图和槽位: 查一下浦发银行最近的公告")
    print(data["intent"], data["slots"])

    # 自定义系统提示
    answer = chat("分析这个公告", system="你是投研分析师")

    # 批量统计
    print(get_usage_stats())
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import LLM
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)


# ---------- 用量统计 ----------
@dataclass
class UsageStats:
    """累计 token 用量。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    _by_model: dict[str, int] = field(default_factory=dict)

    def add(self, usage: dict | None, latency_ms: float, model: str) -> None:
        self.calls += 1
        self.total_latency_ms += latency_ms
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
            self.total_tokens += usage.get("total_tokens", 0) or 0
        self._by_model[model] = self._by_model.get(model, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        avg = self.total_latency_ms / self.calls if self.calls else 0
        return {
            "调用次数": self.calls,
            "失败次数": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "平均延迟_ms": round(avg, 1),
            "按模型": dict(self._by_model),
        }


_usage = UsageStats()


def get_usage_stats() -> dict[str, Any]:
    """获取累计用量。"""
    return _usage.as_dict()


def reset_usage_stats() -> None:
    """重置用量统计。"""
    global _usage
    _usage = UsageStats()


# ---------- 异常 ----------
class LLMError(RuntimeError):
    """LLM 调用异常。"""


class LLMConfigError(LLMError):
    """配置缺失。"""


# ---------- 核心调用 ----------
def _check_config() -> None:
    """检查配置是否完整。"""
    if not LLM["api_key"]:
        raise LLMConfigError(
            "缺少 QIANWEN_API_KEY。请设置环境变量：\n"
            "  $env:QIANWEN_API_KEY='sk-xxx'\n"
            "注意：如果变量已在用户级设置，需要重启终端才能生效。"
        )
    if not LLM["base_url"]:
        raise LLMConfigError("缺少 QIANWEN_BASE_URL")


def _clean_json_text(text: str) -> str:
    """清洗 LLM 返回的 JSON 文本。

    常见问题：
    - 被 ```json ... ``` 包裹
    - 前后有多余文字
    - 尾部有逗号
    """
    t = text.strip()

    # 去掉 markdown 代码块
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # 提取第一个 { 到最后一个 } （处理前后有解释文字的情况）
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start : end + 1]

    # 去掉对象/数组尾部多余逗号
    t = re.sub(r",\s*([}\]])", r"\1", t)

    return t.strip()


def chat(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 3,
    timeout: int | None = None,
) -> str:
    """调用 LLM 返回文本。

    参数：
        prompt: 用户输入
        system: 系统提示（可选）
        model: 覆盖默认模型
        temperature: 覆盖默认温度
        max_tokens: 覆盖默认最大 token
        retries: 重试次数
        timeout: 超时秒数

    返回：模型回复的纯文本
    """
    _check_config()

    mdl = model or LLM["model"]
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": temperature if temperature is not None else LLM["temperature"],
        "max_tokens": max_tokens if max_tokens is not None else LLM["max_tokens"],
    }

    url = f"{LLM['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM['api_key']}",
        "Content-Type": "application/json",
    }

    prompt_chars = len(prompt) + (len(system) if system else 0)
    logger.debug(
        "LLM 请求 model=%s prompt_chars=%d max_tokens=%d",
        mdl,
        prompt_chars,
        payload["max_tokens"],
    )

    last_err: Exception | None = None
    t_start = time.monotonic()

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout or LLM["timeout"],
            )

            latency_ms = (time.monotonic() - t_start) * 1000

            if resp.status_code == 429:
                wait = 5 * attempt
                logger.warning("LLM 限流 429，等待 %ds 后重试 (%d/%d)", wait, attempt, retries)
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                body = resp.text[:300]
                logger.error("LLM HTTP %d: %s", resp.status_code, body)
                # 4xx 客户端错误不重试（除 429）
                if 400 <= resp.status_code < 500:
                    raise LLMError(f"LLM 请求失败 HTTP {resp.status_code}: {body}")
                last_err = LLMError(f"HTTP {resp.status_code}: {body}")
                time.sleep(2**attempt)
                continue

            data = resp.json()

            if not data.get("choices"):
                raise LLMError(f"响应缺少 choices: {str(data)[:200]}")

            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage")
            _usage.add(usage, latency_ms, mdl)

            logger.debug(
                "LLM 响应 耗时=%.0fms reply_chars=%d usage=%s",
                latency_ms,
                len(content),
                usage,
            )

            return content.strip()

        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            _usage.failed_calls += 1
            if attempt < retries:
                backoff = 2**attempt
                logger.warning(
                    "LLM 调用失败 (%d/%d)，%ds 后重试: %s", attempt, retries, backoff, e
                )
                time.sleep(backoff)
            else:
                logger.error("LLM 调用最终失败: %s", e)

    raise LLMError(f"LLM 调用失败（已重试 {retries} 次）: {last_err}")


def chat_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    retries: int = 3,
    default: Any = None,
) -> Any:
    """调用 LLM 并解析为 JSON。

    会自动在系统提示中追加 JSON 输出要求，并清洗返回值。

    参数：
        default: 解析失败时的返回值（None 则抛出异常）

    返回：解析后的 dict / list
    """
    json_system = (
        (system + "\n\n" if system else "")
        + "请只输出 JSON，不要任何解释文字、不要 markdown 代码块标记。"
    )

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = chat(prompt, system=json_system, model=model, retries=2)
            cleaned = _clean_json_text(raw)

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON 解析失败 (%d/%d): %s | 原始文本前200字: %s",
                    attempt,
                    retries,
                    e,
                    cleaned[:200],
                )
                last_err = e
                # 重试时在提示中强调格式
                prompt = prompt + "\n\n注意：上次输出不是合法 JSON，请严格输出 JSON 格式。"

        except LLMError as e:
            last_err = e
            if attempt >= retries:
                break

    if default is not None:
        logger.warning("JSON 解析最终失败，返回默认值")
        return default

    raise LLMError(f"LLM 未返回合法 JSON: {last_err}")


def chat_stream(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
):
    """流式调用（生成器，逐块 yield 文本）。

    用法：
        for chunk in chat_stream("写一段分析"):
            print(chunk, end="", flush=True)
    """
    _check_config()

    mdl = model or LLM["model"]
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": LLM["temperature"],
        "max_tokens": LLM["max_tokens"],
        "stream": True,
    }

    url = f"{LLM['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM['api_key']}",
        "Content-Type": "application/json",
    }

    logger.debug("LLM 流式请求 model=%s", mdl)

    with requests.post(
        url, headers=headers, json=payload, timeout=LLM["timeout"], stream=True
    ) as resp:
        if resp.status_code != 200:
            raise LLMError(f"流式请求失败 HTTP {resp.status_code}: {resp.text[:200]}")

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
            except json.JSONDecodeError:
                continue


def health_check() -> bool:
    """健康检查：发一个最小请求验证配置可用。"""
    try:
        reply = chat("回复OK", max_tokens=10, retries=1)
        logger.info("LLM 健康检查通过 模型=%s 回复=%s", LLM["model"], reply[:20])
        return True
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "LLM 健康检查失败")
        return False


if __name__ == "__main__":
    # 自测
    from utils.logger import log_banner, log_summary

    log_banner(logger, "千问 LLM 客户端自测")
    log_summary(
        logger,
        "配置:",
        {
            "模型": LLM["model"],
            "地址": LLM["base_url"],
            "Key": "(已设置)" if LLM["api_key"] else "(未设置)",
            "温度": LLM["temperature"],
            "最大token": LLM["max_tokens"],
        },
    )

    logger.info("")
    logger.info("[1] 普通对话")
    try:
        reply = chat("用一句话解释什么是股票回购，不超过30字")
        logger.info("  回复: %s", reply)
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "对话失败")

    logger.info("")
    logger.info("[2] JSON 输出")
    try:
        data = chat_json(
            '把这句话拆成意图和槽位：查一下浦发银行最近一个月的回购公告。'
            '输出格式：{"intent": "...", "slots": {"company": "...", "time_range": "...", "type": "..."}}'
        )
        logger.info("  解析结果: %s", json.dumps(data, ensure_ascii=False))
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "JSON 解析失败")

    logger.info("")
    log_summary(logger, "用量统计:", get_usage_stats())
