"""MySQL 连接与建表工具。

用法：
    python utils/db.py init          # 建库建表
    python utils/db.py check         # 检查连接
    python utils/db.py stats         # 统计当前数据量
    python utils/db.py drop          # 删除数据库（危险）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymysql
from pymysql.cursors import DictCursor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import MYSQL, ROOT
from utils.logger import get_logger, log_banner, log_exception, log_summary

logger = get_logger(__name__)


def connect(with_db: bool = True) -> pymysql.connections.Connection:
    """建立 MySQL 连接。

    with_db=False 时不指定数据库（用于建库）。
    """
    cfg = dict(MYSQL)
    if not with_db:
        cfg.pop("database", None)

    logger.debug(
        "连接 MySQL %s:%s/%s (user=%s)",
        cfg["host"],
        cfg["port"],
        cfg.get("database", "(无)"),
        cfg["user"],
    )

    return pymysql.connect(
        **cfg,
        cursorclass=DictCursor,
        autocommit=False,
        connect_timeout=10,
    )


def check() -> bool:
    """检查连接并打印版本。"""
    logger.info("检查 MySQL 连接 ...")
    try:
        conn = connect(with_db=False)
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION() AS v")
            ver = cur.fetchone()["v"]
        conn.close()

        log_summary(
            logger,
            "连接成功:",
            {
                "版本": ver,
                "地址": f"{MYSQL['host']}:{MYSQL['port']}",
                "用户": MYSQL["user"],
                "数据库": MYSQL["database"],
            },
        )
        return True
    except pymysql.err.OperationalError as e:
        code = e.args[0] if e.args else None
        if code == 1045:
            logger.error(
                "认证失败 (1045)。请检查 MYSQL_PASSWORD 环境变量是否设置正确"
            )
            logger.info("  当前密码: %s", "(空)" if not MYSQL["password"] else "(已设置)")
            logger.info("  设置方式: $env:MYSQL_PASSWORD='你的密码'")
        elif code == 2003:
            logger.error("无法连接 (2003)。请确认 MySQL 服务已启动")
            logger.info("  启动方式: net start MySQL80")
        else:
            log_exception(logger, e, "MySQL 连接失败")
        return False
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "MySQL 连接失败")
        return False


def init_schema() -> bool:
    """执行 schema.sql 建库建表。"""
    schema_fp = ROOT / "sql" / "schema.sql"
    if not schema_fp.exists():
        logger.error("找不到 %s", schema_fp)
        return False

    logger.info("读取 %s", schema_fp)
    sql_text = schema_fp.read_text(encoding="utf-8")

    # 建库阶段不能指定 database
    conn = connect(with_db=False)
    executed = 0
    try:
        with conn.cursor() as cur:
            # 按分号切分语句，跳过空行与纯注释
            statements = []
            buf = []
            for line in sql_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("--") or not stripped:
                    continue
                buf.append(line)
                if stripped.endswith(";"):
                    stmt = "\n".join(buf).strip().rstrip(";")
                    if stmt:
                        statements.append(stmt)
                    buf = []

            logger.info("解析出 %d 条 SQL 语句", len(statements))

            for i, stmt in enumerate(statements, 1):
                first_word = stmt.split()[0].upper() if stmt.split() else "?"
                logger.debug("[%d/%d] %s ...", i, len(statements), first_word)
                cur.execute(stmt)
                executed += 1

        conn.commit()
        logger.info("建库建表完成，共执行 %d 条语句", executed)
        return True

    except Exception as e:  # noqa: BLE001
        conn.rollback()
        log_exception(logger, e, f"建表失败（已执行 {executed} 条）")
        return False
    finally:
        conn.close()


def stats() -> None:
    """统计各表数据量。"""
    tables = [
        "announcements",
        "companies",
        "quotes",
        "fx_rates",
        "gold_prices",
        "etl_jobs",
    ]

    try:
        conn = connect()
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "无法连接数据库")
        return

    try:
        with conn.cursor() as cur:
            logger.info("")
            logger.info("  %-20s %12s", "表名", "行数")
            logger.info("  " + "-" * 34)

            for t in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) AS c FROM {t}")
                    c = cur.fetchone()["c"]
                    logger.info("  %-20s %12s", t, f"{c:,}")
                except Exception:  # noqa: BLE001
                    logger.info("  %-20s %12s", t, "(不存在)")

            # 公告分层统计
            try:
                cur.execute(
                    "SELECT level, COUNT(*) AS c FROM announcements GROUP BY level ORDER BY level"
                )
                rows = cur.fetchall()
                if rows:
                    logger.info("")
                    logger.info("  公告分层:")
                    for r in rows:
                        logger.info("    %s: %10s", r["level"], f"{r['c']:,}")
            except Exception:  # noqa: BLE001
                pass

            # 时间范围
            try:
                cur.execute(
                    "SELECT MIN(pub_date) a, MAX(pub_date) b FROM announcements"
                )
                r = cur.fetchone()
                if r and r["a"]:
                    logger.info("")
                    logger.info("  公告时间范围: %s ~ %s", r["a"], r["b"])
            except Exception:  # noqa: BLE001
                pass
    finally:
        conn.close()


def drop() -> bool:
    """删除数据库（危险操作，需二次确认）。"""
    db = MYSQL["database"]
    logger.warning("=" * 60)
    logger.warning("  危险操作：即将删除数据库 %s", db)
    logger.warning("=" * 60)

    answer = input(f"确认删除 {db}？输入 yes 继续: ").strip().lower()
    if answer != "yes":
        logger.info("已取消")
        return False

    try:
        conn = connect(with_db=False)
        with conn.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS `{db}`")
        conn.commit()
        conn.close()
        logger.info("数据库 %s 已删除", db)
        return True
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "删除数据库失败")
        return False


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"

    try:
        if action == "check":
            sys.exit(0 if check() else 1)
        elif action == "init":
            sys.exit(0 if init_schema() else 1)
        elif action == "stats":
            stats()
        elif action == "drop":
            sys.exit(0 if drop() else 1)
        else:
            logger.error("未知命令: %s", action)
            logger.info("用法: python utils/db.py [check|init|stats|drop]")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("用户中断")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "命令执行异常")
        sys.exit(1)
