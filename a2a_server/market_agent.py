"""市场数据 Agent（端口 5007）。

职责：查询行情、汇率、黄金等市场数据。

风格：轻量自研（FastAPI 直连数据源），无框架依赖。

能力：
- 实时指数行情（新浪财经）
- 人民币汇率中间价（中国货币网）
- 黄金行情（上海黄金交易所）
- 市场概览（三者汇总）

数据全部来自本地 MySQL（已采集入库），不实时请求外部接口，
保证响应速度和稳定性。

⚠️ 踩坑记录：
    MySQL 的 DECIMAL 返回 Decimal、DATE 返回 date，不能直接格式化或序列化，
    必须先转成 float / str。
"""
from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a2a.server.agent_execution import RequestContext

from a2a_server.base import PORTS, BaseExecutor, make_agent_card, run_server
from utils.db import connect
from utils.logger import get_logger

logger = get_logger(__name__)


def _f(v: Any, default: float = 0.0) -> float:
    """安全转 float（处理 Decimal / None / str）。"""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _s(v: Any) -> str:
    """安全转 str（处理 date / datetime / Decimal）。"""
    if v is None:
        return ""
    return str(v)


class MarketExecutor(BaseExecutor):
    """市场数据 Executor。"""

    agent_name = "market"

    # 意图关键词
    KEYWORDS = {
        "index": ["指数", "大盘", "行情", "上证", "深证", "创业板", "沪深300", "科创", "股市"],
        "fx": ["汇率", "人民币", "美元", "欧元", "日元", "外汇", "中间价", "换汇"],
        "gold": ["黄金", "金价", "贵金属", "Au99"],
        "overview": ["市场", "概览", "整体", "今天怎么样", "综合"],
    }

    def _detect_intent(self, query: str) -> list[str]:
        """识别查询意图（可多选）。"""
        found = []
        for intent, kws in self.KEYWORDS.items():
            if any(kw in query for kw in kws):
                found.append(intent)

        if not found:
            found = ["overview"]

        # 明确要概览时，不再重复返回细分
        if "overview" in found and len(found) > 1:
            found.remove("overview")

        return found

    def _get_index(self) -> list[dict]:
        """查最新指数行情。"""
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT sec_code, sec_name, price, change_pct, change_amt,
                           quote_date, quote_time
                    FROM quotes
                    ORDER BY quote_date DESC, sec_code
                    LIMIT 10
                """)
                return list(cur.fetchall())
        finally:
            conn.close()

    def _get_fx(self, limit: int = 8) -> list[dict]:
        """查最新汇率（主要货币优先）。

        ⚠️ 踩坑：SQL 里用 Python 参数化（%s）时，LIKE 的字面量 % 必须写成 %%，
        否则 pymysql 做 `query % args` 时会把 `%美` 当成格式符，
        报 "unsupported format character '?' (0x7f8e)"。
        """
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pair, pair_en, rate, change_bp, rate_date
                    FROM fx_rates
                    WHERE rate_date = (SELECT MAX(rate_date) FROM fx_rates)
                    ORDER BY
                        CASE
                            WHEN pair LIKE '%%美元%%' THEN 1
                            WHEN pair LIKE '%%欧元%%' THEN 2
                            WHEN pair LIKE '%%日元%%' THEN 3
                            WHEN pair LIKE '%%港元%%' THEN 4
                            WHEN pair LIKE '%%英镑%%' THEN 5
                            ELSE 9
                        END,
                        pair
                    LIMIT %s
                """, (limit,))
                return list(cur.fetchall())
        finally:
            conn.close()

    def _get_gold(self, days: int = 5) -> list[dict]:
        """查最近 N 天黄金行情。"""
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT trade_date, contract, open_price, high_price,
                           low_price, close_price
                    FROM gold_prices
                    ORDER BY trade_date DESC
                    LIMIT %s
                """, (days,))
                return list(reversed(list(cur.fetchall())))
        finally:
            conn.close()

    async def handle(self, query: str, context: RequestContext) -> dict[str, Any]:
        """处理市场数据查询。"""
        intents = self._detect_intent(query)
        self.logger.info("识别意图: %s", intents)

        def _do_query():
            result = {}
            if "index" in intents or "overview" in intents:
                result["index"] = self._get_index()
            if "fx" in intents or "overview" in intents:
                result["fx"] = self._get_fx()
            if "gold" in intents or "overview" in intents:
                result["gold"] = self._get_gold()
            return result

        data = await asyncio.to_thread(_do_query)

        lines: list[str] = []
        sources: list[str] = []

        # ---- 指数 ----
        if data.get("index"):
            lines.append("【主要指数】")
            for q in data["index"]:
                pct = q.get("change_pct")
                pct_f = _f(pct, default=0.0) if pct is not None else None
                if pct_f is None:
                    arrow, pct_s = "—", "N/A"
                elif pct_f > 0:
                    arrow, pct_s = "↑", f"+{pct_f:.2f}%"
                elif pct_f < 0:
                    arrow, pct_s = "↓", f"{pct_f:.2f}%"
                else:
                    arrow, pct_s = "—", "0.00%"

                name = _s(q.get("sec_name"))
                price = _f(q.get("price"))
                lines.append(f"  {name:<10} {price:>10,.2f}   {arrow} {pct_s:>8}")
            qt = _s(data["index"][0].get("quote_time"))
            if qt:
                lines.append(f"  （数据时间：{qt}）")
            lines.append("")
            sources.append("指数行情来自新浪财经")

        # ---- 汇率 ----
        if data.get("fx"):
            lines.append("【人民币汇率中间价】")
            for row in data["fx"]:
                pair = _s(row.get("pair"))
                rate = _f(row.get("rate"))
                bp = row.get("change_bp")
                bp_s = f"{_f(bp):+.1f}bp" if bp is not None else "N/A"
                lines.append(f"  {pair:<18} {rate:>10.4f}   {bp_s:>10}")
            rd = _s(data["fx"][0].get("rate_date"))
            if rd:
                lines.append(f"  （报价日期：{rd}）")
            lines.append("")
            sources.append("汇率来自中国货币网")

        # ---- 黄金 ----
        if data.get("gold"):
            lines.append("【黄金行情（Au99.99）】")
            for row in data["gold"]:
                td = _s(row.get("trade_date"))
                close = _f(row.get("close_price"))
                high = _f(row.get("high_price"))
                low = _f(row.get("low_price"))
                lines.append(
                    f"  {td}   收盘 {close:>8.2f}   "
                    f"高 {high:>8.2f}   低 {low:>8.2f}"
                )
            lines.append("")
            sources.append("黄金行情来自上海黄金交易所")

        if not lines:
            return {
                "answer": f"没有查到与「{query}」相关的市场数据。",
                "data": {"query": query, "intents": intents},
                "sources": [],
            }

        # ---- 结构化数据（先做类型转换，避免 Decimal/date 序列化失败）----
        detail = {}
        if data.get("index"):
            detail["index"] = [
                {
                    "code": _s(r.get("sec_code")),
                    "name": _s(r.get("sec_name")),
                    "price": _f(r.get("price")),
                    "change_pct": _f(r.get("change_pct")),
                    "quote_date": _s(r.get("quote_date")),
                    "quote_time": _s(r.get("quote_time")),
                }
                for r in data["index"]
            ]
        if data.get("fx"):
            detail["fx"] = [
                {
                    "pair": _s(r.get("pair")),
                    "pair_en": _s(r.get("pair_en")),
                    "rate": _f(r.get("rate")),
                    "change_bp": _f(r.get("change_bp")) if r.get("change_bp") is not None else None,
                    "rate_date": _s(r.get("rate_date")),
                }
                for r in data["fx"]
            ]
        if data.get("gold"):
            detail["gold"] = [
                {
                    "trade_date": _s(r.get("trade_date")),
                    "contract": _s(r.get("contract")),
                    "open": _f(r.get("open_price")),
                    "high": _f(r.get("high_price")),
                    "low": _f(r.get("low_price")),
                    "close": _f(r.get("close_price")),
                }
                for r in data["gold"]
            ]

        return {
            "answer": "\n".join(lines).strip(),
            "data": {"query": query, "intents": intents, "detail": detail},
            "sources": sources,
        }


def create_card(port: int = PORTS["market"]):
    return make_agent_card(
        name="市场数据助手",
        description=(
            "查询 A 股主要指数行情、人民币汇率中间价、黄金行情。"
            "数据来自新浪财经、中国货币网、上海黄金交易所的公开数据。"
        ),
        url=f"http://127.0.0.1:{port}/",
        skills=[
            {
                "id": "query_index",
                "name": "指数行情",
                "description": "查询上证指数、深证成指、创业板指等主要指数的最新行情",
                "tags": ["行情", "指数", "A股"],
                "examples": ["今天大盘怎么样", "上证指数多少点"],
            },
            {
                "id": "query_fx",
                "name": "人民币汇率",
                "description": "查询人民币对美元、欧元、日元等货币的中间价",
                "tags": ["汇率", "外汇"],
                "examples": ["人民币兑美元汇率", "今天汇率多少"],
            },
            {
                "id": "query_gold",
                "name": "黄金行情",
                "description": "查询上海黄金交易所 Au99.99 的行情",
                "tags": ["黄金", "贵金属"],
                "examples": ["黄金价格", "金价走势"],
            },
            {
                "id": "market_overview",
                "name": "市场概览",
                "description": "汇总指数、汇率、黄金的市场整体情况",
                "tags": ["概览", "综合"],
                "examples": ["今天市场整体情况", "市场概览"],
            },
        ],
    )


def main() -> None:
    port = PORTS["market"]
    card = create_card(port)
    executor = MarketExecutor()
    run_server(card, executor, port=port)


if __name__ == "__main__":
    main()
