"""多智能体评估框架。

评估维度（参考学术界的 multi-agent 评测标准）：

| 指标 | 说明 |
|---|---|
| TSR (Task Success Rate) | 任务成功率：答案是否包含预期关键信息 |
| Intent Accuracy | 意图识别准确率：选中的 agent 是否与预期一致 |
| Agent Selection | agent 选择准确性（集合匹配） |
| Latency | 端到端延迟（P50/P95/max） |
| Error Rate | 错误率 |

数据集设计原则：
- **可验证**：每条用例的期望值能从数据库/API 直接核对
- **覆盖多意图**：单意图、双意图、三意图都有
- **包含边界**：缺失信息、超出范围、模糊查询

用法：
    python scripts/evaluate.py                    # 跑全部
    python scripts/evaluate.py --category single  # 只跑单意图
    python scripts/evaluate.py --case id          # 跑单条
    python scripts/evaluate.py --no-llm           # 跳过 LLM 相关（省额度）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a_server.orchestrator import Orchestrator
from utils.logger import get_logger, log_banner, log_exception, log_summary

logger = get_logger(__name__)

# 数据集在 tests/ 目录（与 scripts/ 同级）
_TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
DATASET_PATH = _TESTS_DIR / "eval_dataset.json"
RESULT_PATH = _TESTS_DIR / "eval_results.json"


def load_dataset() -> list[dict]:
    """加载评估数据集。"""
    if not DATASET_PATH.exists():
        logger.error("找不到数据集: %s", DATASET_PATH)
        return []
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def check_keywords(answer: str, keywords: list[str], mode: str = "all") -> tuple[bool, list[str]]:
    """检查答案是否包含关键词。

    参数：
        mode: "all" 全部命中 | "any" 任一命中

    返回：(是否通过, 命中的关键词列表)
    """
    if not keywords:
        return True, []

    hit = [kw for kw in keywords if kw in answer]

    if mode == "any":
        return len(hit) > 0, hit
    return len(hit) == len(keywords), hit


def check_forbidden(answer: str, forbidden: list[str]) -> tuple[bool, list[str]]:
    """检查答案是否包含禁止内容（如幻觉标记）。"""
    if not forbidden:
        return True, []
    hit = [f for f in forbidden if f in answer]
    return len(hit) == 0, hit


def run_case(orch: Orchestrator, case: dict) -> dict[str, Any]:
    """执行单条评估用例。"""
    qid = case["id"]
    query = case["query"]
    expect = case.get("expect", {})

    t0 = time.time()
    try:
        result = orch.run(query)
        elapsed = time.time() - t0
        answer = result.get("answer", "")
        intents = result.get("intents", [])
        error = None
    except Exception as e:  # noqa: BLE001
        elapsed = time.time() - t0
        answer = ""
        intents = []
        error = f"{type(e).__name__}: {e}"
        log_exception(logger, e, f"用例 {qid} 执行失败")

    # ---- 评估各项指标 ----
    metrics: dict[str, Any] = {}

    # 1. 意图准确性
    expect_intents = set(expect.get("intents", []))
    actual_intents = set(intents)
    metrics["intent_match"] = expect_intents == actual_intents if expect_intents else True
    metrics["expected_intents"] = sorted(expect_intents)
    metrics["actual_intents"] = sorted(actual_intents)

    # 2. 关键词覆盖
    kw_mode = expect.get("keyword_mode", "all")
    kw_ok, kw_hit = check_keywords(answer, expect.get("keywords", []), kw_mode)
    metrics["keywords_ok"] = kw_ok
    metrics["keywords_hit"] = kw_hit
    metrics["keywords_expect"] = expect.get("keywords", [])

    # 3. 禁止内容
    fb_ok, fb_hit = check_forbidden(answer, expect.get("forbidden", []))
    metrics["forbidden_ok"] = fb_ok
    metrics["forbidden_hit"] = fb_hit

    # 4. 追问场景
    if expect.get("need_follow_up"):
        metrics["follow_up_ok"] = result.get("need_follow_up", False) if not error else False
    else:
        metrics["follow_up_ok"] = True

    # 5. 综合判定
    success = (
        error is None
        and metrics["intent_match"]
        and kw_ok
        and fb_ok
        and metrics["follow_up_ok"]
    )

    return {
        "id": qid,
        "query": query,
        "category": case.get("category", "unknown"),
        "success": success,
        "error": error,
        "elapsed": round(elapsed, 2),
        "answer_len": len(answer),
        "answer_preview": answer[:300],
        "metrics": metrics,
    }


def summarize(results: list[dict]) -> dict[str, Any]:
    """汇总评估结果。"""
    if not results:
        return {}

    n = len(results)
    ok = sum(1 for r in results if r["success"])
    errs = sum(1 for r in results if r["error"])

    latencies = sorted(r["elapsed"] for r in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    lat_max = latencies[-1]
    avg_lat = sum(latencies) / n

    intent_ok = sum(1 for r in results if r["metrics"]["intent_match"])
    kw_ok = sum(1 for r in results if r["metrics"]["keywords_ok"])
    fb_ok = sum(1 for r in results if r["metrics"]["forbidden_ok"])

    # 按类别统计
    by_cat: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "ok": 0}
        by_cat[cat]["total"] += 1
        if r["success"]:
            by_cat[cat]["ok"] += 1

    return {
        "total": n,
        "success": ok,
        "tsr": round(ok / n * 100, 1),
        "error_rate": round(errs / n * 100, 1),
        "intent_accuracy": round(intent_ok / n * 100, 1),
        "keyword_coverage": round(kw_ok / n * 100, 1),
        "forbidden_pass": round(fb_ok / n * 100, 1),
        "latency_avg": round(avg_lat, 2),
        "latency_p50": round(p50, 2),
        "latency_p95": round(p95, 2),
        "latency_max": round(lat_max, 2),
        "by_category": {
            k: {"total": v["total"], "ok": v["ok"], "rate": round(v["ok"] / v["total"] * 100, 1)}
            for k, v in by_cat.items()
        },
    }


def print_results(results: list[dict], summary: dict) -> None:
    """打印详细结果。"""
    logger.info("")
    log_banner(logger, "逐条结果")
    logger.info(
        "  %-22s %-8s %-6s %-8s %-8s %s",
        "用例ID", "类别", "状态", "意图", "耗时", "说明",
    )
    logger.info("  " + "-" * 88)

    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        m = r["metrics"]
        intent_mark = "OK" if m["intent_match"] else f"X({','.join(m['actual_intents']) or '无'})"
        kw_mark = "" if m["keywords_ok"] else f" 缺关键词: {m['keywords_expect']}"
        err_mark = f" ERR:{r['error'][:30]}" if r["error"] else ""

        logger.info(
            "  %-22s %-8s %-6s %-8s %6.1fs %s%s",
            r["id"],
            r["category"],
            status,
            intent_mark,
            r["elapsed"],
            kw_mark,
            err_mark,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="多智能体评估")
    p.add_argument("--category", help="只跑指定类别")
    p.add_argument("--case", help="只跑指定用例 ID")
    p.add_argument("--no-llm", action="store_true", help="跳过需要 LLM 的用例")
    args = p.parse_args()

    log_banner(logger, "多智能体评估")

    dataset = load_dataset()
    if not dataset:
        sys.exit(1)

    # 过滤
    cases = dataset
    if args.case:
        cases = [c for c in dataset if c["id"] == args.case]
    elif args.category:
        cases = [c for c in dataset if c.get("category") == args.category]
    if args.no_llm:
        cases = [c for c in cases if not c.get("need_llm")]

    if not cases:
        logger.error("没有匹配的用例")
        sys.exit(1)

    logger.info("待评估用例: %d 条", len(cases))
    logger.info("")

    orch = Orchestrator()

    # 检查服务可用性
    cards = orch.discover_agents()
    if not cards:
        logger.error("没有可用的 agent 服务，请先启动：python scripts/run_agents.py")
        sys.exit(1)
    logger.info("可用 agent: %s", list(cards))
    logger.info("")

    results = []
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        logger.info("[%d/%d] %s: %s", i, len(cases), case["id"], case["query"][:50])
        r = run_case(orch, case)
        results.append(r)
        status = "PASS" if r["success"] else "FAIL"
        logger.info("       %s  %.1fs  %s", status, r["elapsed"], r["metrics"]["actual_intents"])

    total_elapsed = time.time() - t_start

    # 汇总
    summary = summarize(results)
    summary["total_elapsed"] = round(total_elapsed, 1)
    summary["finished_at"] = datetime.now().isoformat()

    logger.info("")
    log_banner(logger, "评估汇总")
    log_summary(
        logger,
        "",
        {
            "用例总数": summary["total"],
            "成功": summary["success"],
            "**TSR**": f"{summary['tsr']}%",
            "**意图准确率**": f"{summary['intent_accuracy']}%",
            "关键词覆盖": f"{summary['keyword_coverage']}%",
            "无幻觉/禁词": f"{summary['forbidden_pass']}%",
            "错误率": f"{summary['error_rate']}%",
            "延迟 avg/P50/P95/max": (
                f"{summary['latency_avg']}s / {summary['latency_p50']}s / "
                f"{summary['latency_p95']}s / {summary['latency_max']}s"
            ),
            "总耗时": f"{summary['total_elapsed']}s",
        },
    )

    logger.info("")
    logger.info("  按类别:")
    for cat, st in summary["by_category"].items():
        logger.info("    %-10s %d/%d  %.0f%%", cat, st["ok"], st["total"], st["rate"])

    print_results(results, summary)

    # 保存：合并历史结果，避免分类别跑时互相覆盖
    history: dict[str, dict] = {}
    if RESULT_PATH.exists():
        try:
            old = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
            for r in old.get("results", []):
                history[r["id"]] = r
        except Exception:  # noqa: BLE001
            pass

    for r in results:
        history[r["id"]] = r

    # 用全部历史重新算一次总汇总
    all_results = list(history.values())
    full_summary = summarize(all_results)
    full_summary["finished_at"] = datetime.now().isoformat()
    full_summary["last_run"] = {
        "category": args.category or ("single case" if args.case else "all"),
        "count": len(results),
        "at": datetime.now().isoformat(),
    }

    RESULT_PATH.write_text(
        json.dumps(
            {"summary": full_summary, "results": all_results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("  累计结果（含历史）")
    logger.info("  用例总数: %d  成功: %d  TSR: %.1f%%",
                full_summary["total"], full_summary["success"], full_summary["tsr"])
    logger.info("  意图准确率: %.1f%%  延迟 P95: %.1fs",
                full_summary["intent_accuracy"], full_summary["latency_p95"])
    logger.info("=" * 60)
    logger.info("结果已保存: %s", RESULT_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("用户中断")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "评估失败")
        sys.exit(1)
