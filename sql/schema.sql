-- 上市公司公告情报系统 —— MySQL 表结构
-- 库名: cn_market
-- 字符集: utf8mb4（支持中文与 emoji）

CREATE DATABASE IF NOT EXISTS cn_market
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE cn_market;

-- ============================================================
-- 1. 公告主表
-- ============================================================
CREATE TABLE IF NOT EXISTS announcements (
  announcement_id   VARCHAR(32)  NOT NULL COMMENT '巨潮公告唯一ID，可上官网核对',
  sec_code          VARCHAR(16)  NOT NULL COMMENT '股票代码',
  sec_name          VARCHAR(128) NOT NULL COMMENT '公司简称',
  title             VARCHAR(1024) NOT NULL COMMENT '公告标题',
  announcement_time BIGINT       NULL     COMMENT '公告时间戳(毫秒)',
  pub_date          DATE         NOT NULL COMMENT '发布日期',
  plate             VARCHAR(8)   NULL     COMMENT '板块 sz/sh/bj',
  adjunct_url       VARCHAR(512) NULL     COMMENT 'PDF 相对路径',
  adjunct_size_kb   INT          NULL     COMMENT 'PDF 大小(KB)',
  adjunct_type      VARCHAR(16)  NULL     COMMENT '附件类型',
  announcement_type VARCHAR(128) NULL     COMMENT '公告分类名',
  level             CHAR(2)      NOT NULL DEFAULT 'L2' COMMENT '重要性分层 L1/L2/L3',
  is_embedded       TINYINT      NOT NULL DEFAULT 0 COMMENT '是否已向量化',
  created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (announcement_id),
  KEY idx_sec_code (sec_code),
  KEY idx_pub_date (pub_date),
  KEY idx_level (level),
  KEY idx_embedded (is_embedded),
  KEY idx_sec_date (sec_code, pub_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='上市公司公告';


-- ============================================================
-- 2. 公司基本信息（来自上交所）
-- ============================================================
CREATE TABLE IF NOT EXISTS companies (
  sec_code        VARCHAR(16)  NOT NULL COMMENT '股票代码',
  sec_name        VARCHAR(128) NOT NULL COMMENT '证券简称',
  company_abbr    VARCHAR(128) NULL     COMMENT '公司简称',
  full_name       VARCHAR(512) NULL     COMMENT '公司全称',
  full_name_en    VARCHAR(512) NULL     COMMENT '英文全称',
  csrc_code       VARCHAR(16)  NULL     COMMENT '证监会行业代码',
  csrc_desc       VARCHAR(128) NULL     COMMENT '证监会行业名称',
  area_name       VARCHAR(64)  NULL     COMMENT '地区代码',
  area_desc       VARCHAR(128) NULL     COMMENT '地区名称',
  list_board      VARCHAR(16)  NULL     COMMENT '上市板块',
  list_date       DATE         NULL     COMMENT '上市日期',
  delist_date     DATE         NULL     COMMENT '退市日期',
  state_code      VARCHAR(16)  NULL     COMMENT '状态代码',
  product_status  VARCHAR(64)  NULL     COMMENT '产品状态（上交所返回含填充空格，实测21字符）',
  market          VARCHAR(8)   NULL     COMMENT '市场 SSE/SZSE/BJ',
  updated_at      TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

  PRIMARY KEY (sec_code),
  KEY idx_csrc (csrc_code),
  KEY idx_area (area_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='上市公司基本信息';


-- ============================================================
-- 3. 实时行情快照（东方财富）
-- ============================================================
CREATE TABLE IF NOT EXISTS quotes (
  sec_code    VARCHAR(16) NOT NULL COMMENT '股票/指数代码',
  sec_name    VARCHAR(128) NULL    COMMENT '名称',
  price       DECIMAL(18,4) NULL   COMMENT '最新价',
  change_pct  DECIMAL(10,4) NULL   COMMENT '涨跌幅(%)',
  change_amt  DECIMAL(18,4) NULL   COMMENT '涨跌额',
  market      VARCHAR(8)  NULL     COMMENT '市场 1=沪 0=深',
  quote_date  DATE        NOT NULL COMMENT '行情日期',
  quote_time  DATETIME    NULL     COMMENT '采集时间',
  created_at  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (sec_code, quote_date),
  KEY idx_quote_date (quote_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='行情快照';


-- ============================================================
-- 4. 人民币汇率中间价（中国货币网）
-- ============================================================
CREATE TABLE IF NOT EXISTS fx_rates (
  pair        VARCHAR(32)   NOT NULL COMMENT '货币对，如 USD/CNY',
  pair_en     VARCHAR(32)   NULL     COMMENT '英文货币对',
  rate        DECIMAL(18,6) NULL     COMMENT '汇率',
  change_bp   DECIMAL(10,2) NULL     COMMENT '变动(基点)',
  rate_date   DATE          NOT NULL COMMENT '报价日期',
  created_at  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (pair, rate_date),
  KEY idx_rate_date (rate_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人民币汇率中间价';


-- ============================================================
-- 5. 黄金行情（上海黄金交易所）
-- ============================================================
CREATE TABLE IF NOT EXISTS gold_prices (
  trade_date  DATE          NOT NULL COMMENT '交易日',
  contract    VARCHAR(32)   NOT NULL DEFAULT 'Au99.99' COMMENT '合约',
  open_price  DECIMAL(18,4) NULL,
  high_price  DECIMAL(18,4) NULL,
  low_price   DECIMAL(18,4) NULL,
  close_price DECIMAL(18,4) NULL,
  volume      BIGINT        NULL,
  created_at  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP,

  PRIMARY KEY (trade_date, contract),
  KEY idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='黄金行情';


-- ============================================================
-- 6. 采集任务日志
-- ============================================================
CREATE TABLE IF NOT EXISTS etl_jobs (
  id          BIGINT      NOT NULL AUTO_INCREMENT,
  job_name    VARCHAR(64) NOT NULL COMMENT '任务名',
  target_date DATE        NULL     COMMENT '处理的数据日期',
  status      VARCHAR(16) NOT NULL COMMENT 'success/failed/running',
  row_count   INT         NULL     COMMENT '处理条数',
  message     VARCHAR(512) NULL    COMMENT '错误信息',
  started_at  TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at TIMESTAMP   NULL,

  PRIMARY KEY (id),
  KEY idx_job_name (job_name),
  KEY idx_target_date (target_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ETL 任务日志';


-- ============================================================
-- 7. 常用视图：按公司聚合公告数
-- ============================================================
CREATE OR REPLACE VIEW v_company_announcement_stats AS
SELECT
  a.sec_code,
  a.sec_name,
  COUNT(*)                                    AS total_count,
  SUM(a.level = 'L1')                         AS l1_count,
  SUM(a.level = 'L2')                         AS l2_count,
  SUM(a.level = 'L3')                         AS l3_count,
  MIN(a.pub_date)                             AS first_date,
  MAX(a.pub_date)                             AS last_date
FROM announcements a
GROUP BY a.sec_code, a.sec_name;


-- ============================================================
-- 8. 常用视图：按日期统计
-- ============================================================
CREATE OR REPLACE VIEW v_daily_stats AS
SELECT
  pub_date,
  COUNT(*)            AS total_count,
  SUM(level = 'L1')   AS l1_count,
  SUM(level = 'L2')   AS l2_count,
  SUM(level = 'L3')   AS l3_count,
  COUNT(DISTINCT sec_code) AS company_count
FROM announcements
GROUP BY pub_date;
