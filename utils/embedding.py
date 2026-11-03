"""向量化后端：千问 API 和 BGE-M3 本地模型。

统一接口：
    backend = get_backend()          # 根据配置自动选择
    vectors = backend.encode(texts)  # list[str] -> list[list[float]]

千问（qwen）：
    - OpenAI 兼容接口，无需额外依赖
    - text-embedding-v4 / qwen3.7-text-embedding
    - 批量上限 10 条（v4）或 20 条（qwen3.7）
    - 价格 0.0005 元/千 token

BGE-M3（bge-m3）：
    - 本地模型，数据不出内网
    - 需要 sentence-transformers
    - 首次运行会下载约 2.3 GB 模型
    - CPU 推理较慢，建议 ONNX + INT8 量化
"""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import EMBEDDING
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """向量化异常。"""


class QuotaExhaustedError(EmbeddingError):
    """额度耗尽。不重试，应立即停止任务。"""


class EmbeddingBackend(ABC):
    """向量化后端基类。"""

    name: str = "base"

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        """把文本列表编码为向量列表。"""

    @property
    @abstractmethod
    def max_batch(self) -> int:
        """单次请求最大条数。"""


# ============================================================
# 千问 API
# ============================================================
class QwenEmbedding(EmbeddingBackend):
    """千问向量化（OpenAI 兼容接口）。"""

    name = "qwen"

    def __init__(self) -> None:
        self.api_key = EMBEDDING["api_key"]
        self.base_url = EMBEDDING["base_url"]
        self.model = EMBEDDING["model"]
        self._dim = EMBEDDING["dim"]

        if not self.api_key:
            raise EmbeddingError(
                "缺少 API Key。请设置环境变量：\n"
                "  $env:QIANWEN_API_KEY='sk-xxx'"
            )

        # 不同模型的批量上限不同
        if "v1" in self.model or "v2" in self.model:
            self._max_batch = 25
        elif "qwen3.7" in self.model or "qwen3" in self.model:
            self._max_batch = 20
        else:  # v3 / v4
            self._max_batch = 10

        logger.debug(
            "QwenEmbedding 初始化 model=%s dim=%d max_batch=%d",
            self.model,
            self._dim,
            self._max_batch,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_batch(self) -> int:
        return self._max_batch

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        if not texts:
            return []

        bs = batch_size or self._max_batch
        bs = min(bs, self._max_batch)

        url = f"{self.base_url.rstrip('/')}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_vectors: list[list[float]] = []
        total = len(texts)

        for start in range(0, total, bs):
            batch = list(texts[start : start + bs])
            # 空字符串会导致接口报错，替换成占位符
            batch = [t if t and t.strip() else "（空）" for t in batch]

            payload = {
                "model": self.model,
                "input": batch,
                "dimensions": self._dim,
                "encoding_format": "float",
            }

            logger.debug(
                "向量化请求 batch=%d/%d size=%d", start // bs + 1, (total + bs - 1) // bs, len(batch)
            )

            vectors = self._post_with_retry(url, headers, payload, len(batch))
            all_vectors.extend(vectors)

        logger.debug("向量化完成 共 %d 条", len(all_vectors))
        return all_vectors

    def _post_with_retry(
        self, url: str, headers: dict, payload: dict, expect: int, retries: int = 3
    ) -> list[list[float]]:
        last_err: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=60)

                if resp.status_code == 429:
                    wait = 5 * attempt
                    logger.warning("向量化限流 429，等待 %ds (%d/%d)", wait, attempt, retries)
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    body = resp.text[:500]

                    # ⚠️ 额度耗尽：不重试，直接抛专用异常让上层停止整个任务
                    if "Free quota exhausted" in body or "AllocationQuota" in body:
                        logger.error("=" * 70)
                        logger.error("  API 免费额度已用尽！")
                        logger.error("  模型: %s", self.model)
                        logger.error("  解决方式:")
                        logger.error("    1. 换模型: $env:EMBEDDING_MODEL='qwen3.7-text-embedding'")
                        logger.error("    2. 或到百炼控制台充值/关闭 '仅使用免费额度' 模式")
                        logger.error("=" * 70)
                        raise QuotaExhaustedError(
                            f"免费额度已用尽（模型 {self.model}）"
                        )

                    logger.error("向量化 HTTP %d: %s", resp.status_code, body)
                    if 400 <= resp.status_code < 500:
                        raise EmbeddingError(f"向量化失败 HTTP {resp.status_code}: {body}")
                    last_err = EmbeddingError(f"HTTP {resp.status_code}")
                    time.sleep(2**attempt)
                    continue

                data = resp.json()
                items = data.get("data") or []
                if len(items) != expect:
                    logger.warning("返回条数不符 期望=%d 实际=%d", expect, len(items))

                items_sorted = sorted(items, key=lambda x: x.get("index", 0))
                return [it["embedding"] for it in items_sorted]

            except (QuotaExhaustedError, EmbeddingError):
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < retries:
                    logger.warning("向量化失败 (%d/%d)，%ds 后重试: %s", attempt, retries, 2**attempt, e)
                    time.sleep(2**attempt)
                else:
                    logger.error("向量化最终失败: %s", e)

        raise EmbeddingError(f"向量化失败（已重试 {retries} 次）: {last_err}")


# ============================================================
# BGE-M3 本地模型
# ============================================================
class BgeM3Embedding(EmbeddingBackend):
    """BGE-M3 本地模型。"""

    name = "bge-m3"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "缺少 sentence-transformers。安装：\n"
                "  pip install sentence-transformers"
            ) from e

        model_path = EMBEDDING["bge_model_path"]
        logger.info("加载 BGE-M3 模型: %s（首次会下载约 2.3 GB）", model_path)

        t0 = time.time()
        try:
            self.model = SentenceTransformer(model_path)
        except Exception as e:  # noqa: BLE001
            log_exception(logger, e, "BGE-M3 加载失败")
            raise EmbeddingError(f"模型加载失败: {e}") from e

        elapsed = time.time() - t0
        self._dim = self.model.get_sentence_embedding_dimension()
        logger.info("BGE-M3 加载完成 耗时=%.1fs dim=%d", elapsed, self._dim)

        # 允许配置覆盖维度
        if EMBEDDING["dim"] and EMBEDDING["dim"] != self._dim:
            logger.warning(
                "配置维度 %d 与模型维度 %d 不一致，使用模型维度",
                EMBEDDING["dim"],
                self._dim,
            )

        self._max_batch = 32

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_batch(self) -> int:
        return self._max_batch

    def encode(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        if not texts:
            return []

        bs = batch_size or self._max_batch
        bs = min(bs, self._max_batch)

        logger.debug("BGE-M3 编码 %d 条 batch_size=%d", len(texts), bs)

        t0 = time.time()
        embeddings = self.model.encode(
            list(texts),
            batch_size=bs,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        elapsed = time.time() - t0

        speed = len(texts) / elapsed if elapsed > 0 else 0
        logger.debug("BGE-M3 编码完成 耗时=%.1fs 速度=%.1f 条/秒", elapsed, speed)

        return [vec.tolist() for vec in embeddings]


# ============================================================
# 工厂
# ============================================================
def get_backend(name: str | None = None) -> EmbeddingBackend:
    """根据配置获取向量化后端。

    参数：
        name: 覆盖配置，可选 "qwen" / "bge-m3"
    """
    backend = (name or EMBEDDING["backend"]).lower()

    if backend == "qwen":
        return QwenEmbedding()
    elif backend in ("bge-m3", "bgem3", "bge"):
        return BgeM3Embedding()
    else:
        raise EmbeddingError(f"未知的向量化后端: {backend}（可选 qwen / bge-m3）")


def health_check() -> bool:
    """健康检查：编码一句测试文本。"""
    try:
        backend = get_backend()
        logger.info("向量化后端: %s (dim=%d, max_batch=%d)", backend.name, backend.dim, backend.max_batch)

        t0 = time.time()
        vecs = backend.encode(["这是一条测试文本，用于验证向量化功能。"])
        elapsed = (time.time() - t0) * 1000

        if not vecs or len(vecs[0]) != backend.dim:
            logger.error("维度不符: 期望 %d 实际 %d", backend.dim, len(vecs[0]) if vecs else 0)
            return False

        logger.info(
            "健康检查通过 维度=%d 耗时=%.0fms 前3维=%s",
            len(vecs[0]),
            elapsed,
            [round(v, 4) for v in vecs[0][:3]],
        )
        return True
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "向量化健康检查失败")
        return False


if __name__ == "__main__":
    from utils.logger import log_banner, log_summary

    log_banner(logger, "向量化后端自测")
    log_summary(
        logger,
        "配置:",
        {
            "后端": EMBEDDING["backend"],
            "模型": EMBEDDING["model"],
            "维度": EMBEDDING["dim"],
            "批量": EMBEDDING["batch_size"],
            "并发": EMBEDDING["concurrency"],
        },
    )
    logger.info("")
    ok = health_check()
    sys.exit(0 if ok else 1)
