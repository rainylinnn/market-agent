"""全局配置。

所有可调参数集中在这里，避免散落在各脚本中。
敏感信息（API Key、数据库密码）从环境变量读取，不写死在代码里。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = DATA_DIR / "pdf"
CHECKPOINT_DIR = DATA_DIR / "checkpoint"
LOG_DIR = ROOT / "logs"

for _d in (RAW_DIR, PDF_DIR, CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------- 数据采集范围 ----------
# 3 个月：覆盖 2026 半年报 + 三季报
DATE_START = os.getenv("CNA_DATE_START", "2026-08-01")
DATE_END = os.getenv("CNA_DATE_END", "2026-10-31")

# 巨潮分页上限固定 30 条（实测 100/300/500 都只返回 30）
PAGE_SIZE = 30
# 请求间隔（秒），避免被限流
REQUEST_INTERVAL = float(os.getenv("CNA_REQUEST_INTERVAL", "0.25"))
MAX_RETRIES = 3

CNINFO_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_REFERER = "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice"
CNINFO_PDF_BASE = "http://static.cninfo.com.cn/"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------- 公告重要性分层 ----------
# L1：核心公告，下载 PDF 全文 + 向量化
CORE_KEYWORDS = [
    "重大合同", "重大资产重组", "业绩预告", "业绩快报",
    "回购", "股权激励", "分红", "权益分派", "利润分配",
    "诉讼", "仲裁", "关联交易", "对外投资", "收购", "并购",
    "增持", "减持", "控制权", "要约", "退市风险", "破产", "重整",
    "重大事项", "停牌", "复牌", "立案", "处罚", "问询",
]

# L3：模板化公告，只存元数据，不向量化
TEMPLATE_PATTERNS = [
    r"监事会.*决议",
    r"董事会.*决议",
    r"股东大会.*通知",
    r"持续督导",
    r"保荐代表人",
    r"季度报告.*提示",
    r"独立董事.*述职",
    r"审计委员会.*履职",
    r"章程.*修订",
    r"制度.*修订",
]


# ---------- MySQL ----------
# 连接参数全部从环境变量读取，代码中不保留任何凭据
MYSQL = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "cn_market"),
    "charset": "utf8mb4",
}


# ---------- Milvus ----------
MILVUS = {
    "host": os.getenv("MILVUS_HOST", "127.0.0.1"),
    "port": os.getenv("MILVUS_PORT", "19530"),
    "collection": os.getenv("MILVUS_COLLECTION", "cn_announcement"),
}

# 向量维度必须与 embedding 模型输出一致，且建表后不可更改
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))


# ---------- Embedding ----------
EMBEDDING = {
    # 可选: "qwen" | "bge-m3"
    "backend": os.getenv("EMBEDDING_BACKEND", "qwen"),
    "model": os.getenv("EMBEDDING_MODEL", "qwen3.7-text-embedding-flash"),
    "batch_size": int(os.getenv("EMBEDDING_BATCH", "20")),
    "concurrency": int(os.getenv("EMBEDDING_CONCURRENCY", "5")),
    "dim": EMBEDDING_DIM,
    # 千问：OpenAI 兼容接口
    # 优先读 QIANWEN_API_KEY / QIANWEN_BASE_URL，兼容旧的 DASHSCOPE_* 变量
    "api_key": os.getenv("QIANWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY", ""),
    "base_url": (
        os.getenv("QIANWEN_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    # BGE-M3 本地模型
    "bge_model_path": os.getenv("BGE_MODEL_PATH", "BAAI/bge-m3"),
}


# ---------- LLM（对话与理解）----------
LLM = {
    "api_key": os.getenv("QIANWEN_API_KEY", ""),
    "base_url": (
        os.getenv("QIANWEN_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    "model": os.getenv("QIANWEN_MODEL", "qwen3.8-27b"),
    "temperature": float(os.getenv("QIANWEN_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("QIANWEN_MAX_TOKENS", "2048")),
    "timeout": int(os.getenv("QIANWEN_TIMEOUT", "120")),
}


# ---------- 采集控制 ----------
# 按天切片，避免分页深度限制
MAX_PAGES_PER_DAY = int(os.getenv("CNA_MAX_PAGES_PER_DAY", "200"))
# 每处理多少条写一次断点
CHECKPOINT_EVERY = 500


def classify_announcement(title: str) -> str:
    """按标题给公告分层。

    返回 "L1"（核心）、"L2"（重要）、"L3"（模板）。
    """
    import re

    t = title or ""

    # 先判 L1：核心关键词
    for kw in CORE_KEYWORDS:
        if kw in t:
            return "L1"

    # 再判 L3：模板化
    for pat in TEMPLATE_PATTERNS:
        if re.search(pat, t):
            return "L3"

    return "L2"


def summary() -> str:
    """打印当前配置摘要，便于确认。"""
    return (
        f"日期范围   : {DATE_START} ~ {DATE_END}\n"
        f"分页大小   : {PAGE_SIZE} 条/页\n"
        f"请求间隔   : {REQUEST_INTERVAL}s\n"
        f"MySQL      : {MYSQL['host']}:{MYSQL['port']}/{MYSQL['database']}\n"
        f"Milvus     : {MILVUS['host']}:{MILVUS['port']}/{MILVUS['collection']}\n"
        f"Embedding  : {EMBEDDING['backend']} / {EMBEDDING['model']} / {EMBEDDING_DIM} 维\n"
        f"批量/并发  : {EMBEDDING['batch_size']} / {EMBEDDING['concurrency']}"
    )
