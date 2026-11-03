"""Milvus 集合管理与检索。

使用 pymilvus 3.x 的 MilvusClient 新 API（ORM 风格已弃用）。

集合结构：
    id            主键（自增）
    announcement_id  巨潮公告 ID（用于回查 MySQL）
    sec_code      股票代码
    sec_name      公司简称
    title         公告标题
    level         重要性分层
    pub_date      发布日期（用于过滤）
    embedding     向量

用法：
    from utils.milvus_client import get_client, ensure_collection, search

    client = get_client()
    ensure_collection(client)
    search(client, "回购", top_k=10)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

from pymilvus import DataType, MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import EMBEDDING_DIM, MILVUS
from utils.logger import get_logger, log_exception, log_summary

logger = get_logger(__name__)


def get_client() -> MilvusClient:
    """建立 Milvus 连接。"""
    uri = f"http://{MILVUS['host']}:{MILVUS['port']}"
    logger.debug("连接 Milvus %s", uri)
    return MilvusClient(uri=uri)


def collection_exists(client: MilvusClient, name: str | None = None) -> bool:
    """检查集合是否存在。"""
    coll = name or MILVUS["collection"]
    try:
        return client.has_collection(coll)
    except Exception as e:  # noqa: BLE001
        logger.warning("检查集合失败: %s", e)
        return False


def ensure_collection(
    client: MilvusClient,
    name: str | None = None,
    dim: int | None = None,
    drop_if_exists: bool = False,
) -> bool:
    """创建集合（如果不存在）。

    参数：
        drop_if_exists: 是否先删除重建（危险，会丢数据）

    返回：True 表示新建，False 表示已存在
    """
    coll = name or MILVUS["collection"]
    d = dim or EMBEDDING_DIM

    if collection_exists(client, coll):
        if drop_if_exists:
            logger.warning("删除已有集合 %s", coll)
            client.drop_collection(coll)
        else:
            # 校验维度是否一致
            try:
                desc = client.describe_collection(coll)
                for f in desc.get("fields", []):
                    if f.get("name") == "embedding":
                        existing_dim = f.get("params", {}).get("dim")
                        if existing_dim and existing_dim != d:
                            logger.error(
                                "维度不匹配！集合=%d 配置=%d。如需重建请用 --rebuild",
                                existing_dim,
                                d,
                            )
                            raise RuntimeError(
                                f"维度不匹配: 集合 {existing_dim} vs 配置 {d}"
                            )
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001
                pass
            logger.info("集合 %s 已存在（dim=%d）", coll, d)
            return False

    logger.info("创建集合 %s dim=%d", coll, d)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("announcement_id", DataType.VARCHAR, max_length=32)
    schema.add_field("sec_code", DataType.VARCHAR, max_length=16)
    schema.add_field("sec_name", DataType.VARCHAR, max_length=128)
    schema.add_field("title", DataType.VARCHAR, max_length=1024)
    schema.add_field("level", DataType.VARCHAR, max_length=4)
    schema.add_field("pub_date", DataType.VARCHAR, max_length=16)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=d)

    # HNSW 索引：召回率高，适合中小规模
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    # 标量字段索引，加速过滤
    index_params.add_index(field_name="sec_code", index_type="INVERTED")
    index_params.add_index(field_name="sec_name", index_type="INVERTED")
    index_params.add_index(field_name="level", index_type="INVERTED")
    index_params.add_index(field_name="pub_date", index_type="INVERTED")

    client.create_collection(
        collection_name=coll,
        schema=schema,
        index_params=index_params,
    )

    logger.info("集合创建完成 %s", coll)
    return True


def insert_batch(
    client: MilvusClient,
    rows: Sequence[dict],
    name: str | None = None,
) -> int:
    """批量插入向量。

    rows 每条需包含：announcement_id, sec_code, sec_name, title, level, pub_date, embedding
    """
    if not rows:
        return 0

    coll = name or MILVUS["collection"]
    res = client.insert(collection_name=coll, data=list(rows))

    count = res.get("insert_count", 0) if isinstance(res, dict) else 0
    logger.debug("插入 %d 条到 %s", count, coll)
    return count


def flush(client: MilvusClient, name: str | None = None) -> None:
    """强制落盘。"""
    coll = name or MILVUS["collection"]
    try:
        client.flush(coll)
        logger.debug("flush 完成 %s", coll)
    except Exception as e:  # noqa: BLE001
        logger.warning("flush 失败: %s", e)


def get_stats(client: MilvusClient, name: str | None = None) -> dict[str, Any]:
    """获取集合统计。"""
    coll = name or MILVUS["collection"]
    try:
        stats = client.get_collection_stats(coll)
        return {
            "集合": coll,
            "行数": f"{stats.get('row_count', 0):,}",
        }
    except Exception as e:  # noqa: BLE001
        return {"集合": coll, "状态": f"查询失败: {e}"}


def search(
    client: MilvusClient,
    query_vector: list[float],
    top_k: int = 10,
    *,
    level: str | None = None,
    sec_code: str | None = None,
    sec_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    name: str | None = None,
) -> list[dict]:
    """向量检索。

    参数：
        query_vector: 查询向量
        top_k: 返回条数
        level: 过滤重要性分层（L1/L2/L3）
        sec_code: 过滤股票代码（精确匹配）
        sec_name: 过滤公司名（前缀匹配，如 "浦发" 匹配 "浦发银行"）
        date_from / date_to: 日期范围过滤（YYYY-MM-DD）

    返回：命中的公告列表（按相似度降序）
    """
    coll = name or MILVUS["collection"]

    # 构建过滤表达式
    exprs = []
    if level:
        exprs.append(f'level == "{level}"')
    if sec_code:
        exprs.append(f'sec_code == "{sec_code}"')
    if sec_name:
        exprs.append(f'sec_name like "{sec_name}%"')
    if date_from:
        exprs.append(f'pub_date >= "{date_from}"')
    if date_to:
        exprs.append(f'pub_date <= "{date_to}"')

    filter_expr = " and ".join(exprs) if exprs else ""

    res = client.search(
        collection_name=coll,
        data=[query_vector],
        limit=top_k,
        filter=filter_expr,
        output_fields=["announcement_id", "sec_code", "sec_name", "title", "level", "pub_date"],
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
    )

    hits = []
    for hit_list in res:
        for h in hit_list:
            entity = h.get("entity", {})
            hits.append(
                {
                    "score": round(h.get("distance", 0), 4),
                    "announcement_id": entity.get("announcement_id"),
                    "sec_code": entity.get("sec_code"),
                    "sec_name": entity.get("sec_name"),
                    "title": entity.get("title"),
                    "level": entity.get("level"),
                    "pub_date": entity.get("pub_date"),
                }
            )
    return hits


def delete_by_ids(
    client: MilvusClient, announcement_ids: Sequence[str], name: str | None = None
) -> int:
    """按公告 ID 删除（用于重跑）。"""
    coll = name or MILVUS["collection"]
    if not announcement_ids:
        return 0
    expr = "announcement_id in [" + ", ".join(f'"{i}"' for i in announcement_ids) + "]"
    res = client.delete(collection_name=coll, filter=expr)
    return res.get("delete_count", 0) if isinstance(res, dict) else 0


def drop_collection(client: MilvusClient, name: str | None = None) -> None:
    """删除集合（危险）。"""
    coll = name or MILVUS["collection"]
    client.drop_collection(coll)
    logger.warning("集合 %s 已删除", coll)


if __name__ == "__main__":
    import argparse

    from utils.logger import log_banner

    p = argparse.ArgumentParser(description="Milvus 集合管理")
    p.add_argument("action", choices=["check", "init", "stats", "drop"], help="操作")
    p.add_argument("--rebuild", action="store_true", help="删除重建")
    args = p.parse_args()

    log_banner(logger, f"Milvus 集合管理  action={args.action}")

    try:
        client = get_client()

        if args.action == "check":
            ok = collection_exists(client)
            log_summary(
                logger,
                "检查结果:",
                {
                    "地址": f"{MILVUS['host']}:{MILVUS['port']}",
                    "集合名": MILVUS["collection"],
                    "是否存在": "是" if ok else "否",
                },
            )

        elif args.action == "init":
            created = ensure_collection(client, drop_if_exists=args.rebuild)
            logger.info("结果: %s", "新建" if created else "已存在")

        elif args.action == "stats":
            log_summary(logger, "统计:", get_stats(client))

        elif args.action == "drop":
            if collection_exists(client):
                drop_collection(client)
            else:
                logger.info("集合不存在，无需删除")

    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "Milvus 操作失败")
        sys.exit(1)
