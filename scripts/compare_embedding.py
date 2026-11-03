"""向量化方案对比实验。

对比不同方法对检索效果和速度的影响：
1. text-embedding-v4 只编码标题
2. text-embedding-v4 编码"公司名+标题"
3. qwen3.7-text-embedding 编码"公司名+标题"
4. 加 query 指令前缀（query: / passage: 风格）

评估方法：
- 用已知关键词的查询，看 Top-K 命中率
- 记录相似度分布
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from utils.config import EMBEDDING
from utils.db import connect
from utils.logger import get_logger, log_banner, log_summary

logger = get_logger(__name__)


def get_api_key() -> str:
    key = EMBEDDING["api_key"]
    if not key:
        raise RuntimeError("缺少 QIANWEN_API_KEY")
    return key


def embed(texts: list[str], model: str, dim: int = 1024) -> list[list[float]]:
    """直接调用 embedding 接口（自动分批）。

    关键约束（实测）：
    - text-embedding-v4: 单批最多 10 条
    - qwen3.7-text-embedding: 单批最多 20 条
    - 单条文本不能为空，否则 400
    """
    # 各模型的批量上限
    if "qwen3.7" in model or "qwen3" in model:
        max_batch = 20
    elif "v1" in model or "v2" in model:
        max_batch = 25
    else:  # v3 / v4
        max_batch = 10

    url = f"{EMBEDDING['base_url'].rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }

    # 清洗：去空、截断过长文本
    cleaned = []
    for t in texts:
        s = (t or "").strip()
        if not s:
            s = "（空文本）"
        cleaned.append(s[:2000])

    all_vecs: list[list[float]] = []

    for start in range(0, len(cleaned), max_batch):
        batch = cleaned[start : start + max_batch]
        payload = {
            "model": model,
            "input": batch,
            "dimensions": dim,
            "encoding_format": "float",
        }

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:  # noqa: BLE001
                err = r.text[:300]
            raise RuntimeError(f"HTTP {r.status_code}: {err}")

        data = r.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        all_vecs.extend([it["embedding"] for it in items])

    return all_vecs


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def load_samples(per_keyword: int = 8) -> list[dict]:
    """从 MySQL 取有代表性的样本。

    每个关键词取 per_keyword 条，保证样本多样性。
    """
    keywords = ["回购", "重组", "减持", "业绩预告", "股权激励", "诉讼", "分红", "停牌"]

    conn = connect()
    samples: list[dict] = []
    seen: set[str] = set()
    try:
        with conn.cursor() as cur:
            for kw in keywords:
                cur.execute(
                    """
                    SELECT announcement_id, sec_code, sec_name, title, level
                    FROM announcements
                    WHERE title LIKE %s AND level = 'L1'
                    ORDER BY RAND()
                    LIMIT %s
                    """,
                    (f"%{kw}%", per_keyword),
                )
                for r in cur.fetchall():
                    if r["announcement_id"] not in seen:
                        seen.add(r["announcement_id"])
                        r["keyword"] = kw
                        samples.append(r)
    finally:
        conn.close()

    return samples


def run_experiment(name: str, doc_texts: list[str], queries: dict[str, str], model: str) -> dict:
    """跑一组实验。"""
    t0 = time.time()
    doc_vecs = embed(doc_texts, model)
    doc_time = time.time() - t0

    t1 = time.time()
    query_vecs = {k: embed([v], model)[0] for k, v in queries.items()}
    query_time = time.time() - t1

    # 对每个查询，计算与所有文档的相似度
    results = {}
    for qk, qv in query_vecs.items():
        sims = [(i, cosine(qv, dv)) for i, dv in enumerate(doc_vecs)]
        sims.sort(key=lambda x: -x[1])
        results[qk] = sims

    return {
        "name": name,
        "doc_vecs": doc_vecs,
        "query_vecs": query_vecs,
        "results": results,
        "doc_time": doc_time,
        "query_time": query_time,
    }


def main() -> None:
    log_banner(logger, "向量化方案对比实验")

    samples = load_samples(per_keyword=6)
    logger.info("样本数: %d 条", len(samples))

    # 打印样本分布
    from collections import Counter
    kw_dist = Counter(s["keyword"] for s in samples)
    logger.info("关键词分布: %s", dict(kw_dist))

    # 三种文档表示
    variants = {
        "A_仅标题": [s["title"] for s in samples],
        "B_公司名+标题": [f"{s['sec_name']}：{s['title']}" for s in samples],
        "C_标题+公司名": [f"{s['title']}（{s['sec_name']}）" for s in samples],
    }

    # 测试查询
    queries = {
        "回购": "公司回购股份的公告",
        "重组": "重大资产重组",
        "减持": "股东减持股份",
        "业绩预告": "业绩预告",
        "股权激励": "股权激励计划",
        "诉讼": "涉及诉讼的公告",
        "分红": "利润分配方案",
        "停牌": "股票停牌",
    }

    models = ["text-embedding-v4", "qwen3.7-text-embedding"]

    all_results = []

    for model in models:
        for vname, doc_texts in variants.items():
            try:
                logger.info("")
                logger.info("=== 模型 %s / 表示 %s ===", model, vname)
                r = run_experiment(f"{model}|{vname}", doc_texts, queries, model)

                # 评估：每个查询的 Top-1 是否包含对应关键词
                hits_top1 = 0
                hits_top3 = 0
                sims_top1 = []
                details = []

                for qk, sims in r["results"].items():
                    top1_idx = sims[0][0]
                    top3_idx = [s[0] for s in sims[:3]]
                    top1_sim = sims[0][1]

                    top1_kw = samples[top1_idx]["keyword"]
                    top3_kws = [samples[i]["keyword"] for i in top3_idx]

                    ok1 = top1_kw == qk
                    ok3 = qk in top3_kws

                    if ok1:
                        hits_top1 += 1
                    if ok3:
                        hits_top3 += 1
                    sims_top1.append(top1_sim)

                    details.append(
                        {
                            "query": qk,
                            "top1_kw": top1_kw,
                            "top1_sim": round(top1_sim, 4),
                            "hit@1": ok1,
                            "hit@3": ok3,
                        }
                    )

                n = len(queries)
                avg_sim = sum(sims_top1) / len(sims_top1)

                logger.info(
                    "  Top1命中: %d/%d (%.0f%%)  Top3命中: %d/%d (%.0f%%)  平均Top1相似度: %.4f  向量化耗时: %.1fs",
                    hits_top1, n, hits_top1 / n * 100,
                    hits_top3, n, hits_top3 / n * 100,
                    avg_sim, r["doc_time"],
                )

                all_results.append(
                    {
                        "model": model,
                        "variant": vname,
                        "hit1": hits_top1,
                        "hit3": hits_top3,
                        "total": n,
                        "avg_sim": round(avg_sim, 4),
                        "doc_time": round(r["doc_time"], 2),
                        "details": details,
                    }
                )

            except Exception as e:  # noqa: BLE001
                logger.error("  %s / %s 失败: %s", model, vname, e)

    # 汇总
    logger.info("")
    log_banner(logger, "实验结果汇总")
    logger.info(
        "  %-32s %8s %8s %10s %10s",
        "模型|表示", "Top1", "Top3", "平均相似度", "耗时",
    )
    logger.info("  " + "-" * 74)

    for r in sorted(all_results, key=lambda x: (-x["hit1"], -x["avg_sim"])):
        logger.info(
            "  %-32s %7.0f%% %7.0f%% %10.4f %9.1fs",
            f"{r['model']}|{r['variant']}",
            r["hit1"] / r["total"] * 100,
            r["hit3"] / r["total"] * 100,
            r["avg_sim"],
            r["doc_time"],
        )

    # 保存详细结果
    out = Path("data") / "embedding_experiment.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("")
    logger.info("详细结果已存: %s", out)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        logger.error("实验失败: %s", e, exc_info=True)
        sys.exit(1)
