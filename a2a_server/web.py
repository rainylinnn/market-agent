"""Web 界面：多智能体投研助手。

技术栈：FastAPI + 原生 HTML/JS（无前端框架、无 CDN，完全离线可用）

界面结构：
    左侧栏：数据规模 / Agent 状态 / 示例问题
    主区域：对话流
    底部  ：输入框

用法：
    python -m a2a_server.web
    # 访问 http://127.0.0.1:8085
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from a2a_server.orchestrator import Orchestrator
from utils.logger import get_logger, log_exception

logger = get_logger(__name__)

app = FastAPI(title="MarketAgent", description="多智能体投研助手")

# 全局编排器（复用连接）
_orch = Orchestrator()

# 统计缓存（避免每次刷新页面都打 MySQL + Milvus）
_stats_cache: dict = {"at": 0.0, "data": None}
_STATS_TTL = 300.0


def _collect_stats() -> dict:
    """采集数据规模（MySQL + Milvus）。"""
    data = {"announcements": 0, "companies": 0, "vectors": 0, "embedded": 0}

    try:
        from utils.db import connect

        conn = connect()
        cur = conn.cursor()  # DictCursor，返回 dict
        for key, sql in (
            ("announcements", "SELECT COUNT(*) AS n FROM announcements"),
            ("companies", "SELECT COUNT(*) AS n FROM companies"),
            ("embedded", "SELECT COUNT(*) AS n FROM announcements WHERE is_embedded = 1"),
        ):
            cur.execute(sql)
            row = cur.fetchone() or {}
            data[key] = int(row.get("n", 0) or 0)
        cur.close()
        conn.close()
    except Exception as e:  # noqa: BLE001
        logger.warning("统计 MySQL 失败: %s", e)

    try:
        from utils.milvus_client import get_client, get_stats

        client = get_client()
        stats = get_stats(client)
        data["vectors"] = int(str(stats.get("行数", "0")).replace(",", ""))
    except Exception as e:  # noqa: BLE001
        logger.warning("统计 Milvus 失败: %s", e)

    return data


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MarketAgent · 多智能体投研助手</title>
<style>
  :root{
    --bg:#08090c;
    --panel:#0d0f14;
    --panel-2:#11141a;
    --line:#1c2028;
    --line-2:#262c36;
    --txt:#e6e9ef;
    --txt-2:#9aa3b0;
    --txt-3:#5f6875;
    --accent:#5b8cff;
    --accent-2:#7aa2ff;
    --green:#3ecf8e;
    --amber:#e3b341;
    --red:#f0705f;
    --radius:14px;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  html,body{height:100%;}
  body{
    font-family:"Inter","Segoe UI","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
    background:var(--bg); color:var(--txt);
    -webkit-font-smoothing:antialiased;
    overflow:hidden;
  }
  body::before{
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background:
      radial-gradient(900px 500px at 15% -10%, rgba(91,140,255,.10), transparent 70%),
      radial-gradient(700px 400px at 95% 0%, rgba(62,207,142,.06), transparent 70%);
  }

  .app{position:relative; z-index:1; display:flex; height:100vh;}

  /* ---------------- 侧栏 ---------------- */
  .sidebar{
    width:268px; flex:0 0 268px;
    background:var(--panel); border-right:1px solid var(--line);
    display:flex; flex-direction:column; overflow-y:auto;
  }
  .brand{
    padding:20px 20px 18px; border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:11px;
  }
  .logo{
    width:32px; height:32px; border-radius:9px; flex:0 0 32px;
    background:linear-gradient(135deg,#5b8cff,#3ecf8e);
    display:flex; align-items:center; justify-content:center;
    font-size:15px; font-weight:700; color:#08090c;
    box-shadow:0 4px 14px rgba(91,140,255,.28);
  }
  .brand-txt h1{font-size:14.5px; font-weight:650; letter-spacing:.2px;}
  .brand-txt p{font-size:10.5px; color:var(--txt-3); margin-top:2px; letter-spacing:.4px;}

  .side-sec{padding:18px 20px; border-bottom:1px solid var(--line);}
  .side-sec:last-child{border-bottom:none;}
  .side-title{
    font-size:10.5px; font-weight:600; letter-spacing:1.1px;
    color:var(--txt-3); text-transform:uppercase; margin-bottom:13px;
  }

  .stat{display:flex; align-items:baseline; justify-content:space-between; padding:5px 0;}
  .stat-k{font-size:12.5px; color:var(--txt-2);}
  .stat-v{
    font-size:13.5px; font-weight:600; color:var(--txt);
    font-variant-numeric:tabular-nums; letter-spacing:.2px;
  }
  .stat-v.dim{color:var(--txt-3); font-weight:500; font-size:12px;}

  .agent{
    display:flex; align-items:center; gap:10px;
    padding:8px 10px; border-radius:9px; margin-bottom:5px;
    border:1px solid transparent; transition:.16s;
  }
  .agent:hover{background:var(--panel-2); border-color:var(--line);}
  .dot{
    width:6.5px; height:6.5px; border-radius:50%; flex:0 0 6.5px;
    background:var(--txt-3); box-shadow:0 0 0 3px rgba(95,104,117,.12);
  }
  .dot.on{background:var(--green); box-shadow:0 0 0 3px rgba(62,207,142,.14);}
  .dot.off{background:var(--red); box-shadow:0 0 0 3px rgba(240,112,95,.14);}
  .agent-name{font-size:12.5px; color:var(--txt-2);}
  .agent-port{
    margin-left:auto; font-size:10.5px; color:var(--txt-3);
    font-variant-numeric:tabular-nums;
  }

  .ex{
    font-size:12.5px; color:var(--txt-2); padding:8px 10px;
    border-radius:9px; cursor:pointer; border:1px solid transparent;
    transition:.16s; line-height:1.45; display:flex; gap:8px;
  }
  .ex:hover{background:var(--panel-2); border-color:var(--line-2); color:var(--txt);}
  .ex-ico{color:var(--accent); flex:0 0 auto; opacity:.8;}

  /* ---------------- 主区 ---------------- */
  .main{flex:1; display:flex; flex-direction:column; min-width:0;}

  .chat{
    flex:1; overflow-y:auto; padding:32px 0 8px;
    scroll-behavior:smooth;
    display:flex; flex-direction:column;
  }
  .chat-inner{
    max-width:820px; width:100%; margin:0 auto; padding:0 32px;
    display:flex; flex-direction:column; flex:1;
  }

  /* 欢迎屏 */
  .welcome{
    padding:0 0 40px; text-align:center;
    display:flex; flex-direction:column; justify-content:center;
    flex:1;
  }
  .welcome-badge{
    display:inline-flex; align-items:center; gap:7px;
    font-size:11px; color:var(--txt-3); letter-spacing:.5px;
    padding:5px 12px; border-radius:20px;
    background:var(--panel-2); border:1px solid var(--line-2);
    margin-bottom:22px; align-self:center;
  }
  .welcome h2{
    font-size:27px; font-weight:600; letter-spacing:-.3px;
    line-height:1.35; margin-bottom:14px;
    background:linear-gradient(180deg,#fff,#a9b4c6);
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent;
  }
  .welcome p{font-size:13.5px; color:var(--txt-3); line-height:1.85;}

  .cards{
    display:grid; grid-template-columns:repeat(3,1fr); gap:11px;
    margin-top:34px; text-align:left;
    align-items:stretch;
  }
  .card{
    padding:15px 15px 16px; border-radius:12px;
    background:var(--panel); border:1px solid var(--line);
    transition:.18s; display:flex; flex-direction:column;
  }
  .card:hover{border-color:var(--line-2); background:var(--panel-2); transform:translateY(-1px);}
  .card-ico{
    font-size:14px; color:var(--accent); margin-bottom:9px; display:block;
    line-height:1;
  }
  .card h3{font-size:12.5px; font-weight:600; color:var(--txt); margin-bottom:5px;}
  .card p{font-size:11.5px; color:var(--txt-3); line-height:1.6;}

  /* 消息 */
  .msg{margin-bottom:26px; animation:rise .32s cubic-bezier(.22,.8,.3,1);}
  @keyframes rise{from{opacity:0; transform:translateY(9px);} to{opacity:1; transform:none;}}

  .msg.user{display:flex; justify-content:flex-end;}
  .msg.user .bubble{
    background:linear-gradient(135deg,#1d3a6b,#22406f);
    border:1px solid #2c4d80; color:#e8eefc;
    border-radius:14px 14px 4px 14px;
    padding:11px 16px; font-size:14px; line-height:1.65;
    max-width:78%; white-space:pre-wrap; word-break:break-word;
  }

  .bot-head{
    display:flex; align-items:center; gap:9px; margin-bottom:11px;
  }
  .bot-ava{
    width:22px; height:22px; border-radius:7px; flex:0 0 22px;
    background:linear-gradient(135deg,#5b8cff,#3ecf8e);
    display:flex; align-items:center; justify-content:center;
    font-size:10px; font-weight:700; color:#08090c;
  }
  .bot-name{font-size:12px; color:var(--txt-2); font-weight:500;}
  .bot-chips{display:flex; gap:6px; margin-left:auto;}
  .chip{
    font-size:10.5px; padding:2.5px 8px; border-radius:7px;
    background:rgba(91,140,255,.10); color:var(--accent-2);
    border:1px solid rgba(91,140,255,.22); letter-spacing:.2px;
  }
  .chip.warn{background:rgba(227,179,65,.10); color:var(--amber); border-color:rgba(227,179,65,.22);}

  .bubble.bot{
    background:var(--panel); border:1px solid var(--line);
    border-radius:4px 14px 14px 14px;
    padding:17px 20px; font-size:13.8px; line-height:1.78;
    color:#d5dae3;
  }
  .bubble.bot p{margin:0 0 2px;}
  .bubble.bot h3,.bubble.bot h4,.bubble.bot h5{
    font-size:13.5px; font-weight:650; color:var(--txt);
    margin:17px 0 9px; padding-left:10px;
    border-left:2.5px solid var(--accent); line-height:1.5;
  }
  .bubble.bot h3:first-child,.bubble.bot h4:first-child{margin-top:0;}
  .bubble.bot h3.ttl{
    font-size:14.5px; border-left:none; padding-left:0;
    margin:0 0 12px; color:#fff; letter-spacing:.2px;
  }
  .bubble.bot b{color:#fff; font-weight:600;}
  .bubble.bot ul{margin:7px 0 9px; padding-left:19px;}
  .bubble.bot li{margin:3px 0;}
  .bubble.bot li.ann{
    list-style:none; margin-left:-19px; display:flex; gap:10px;
    align-items:baseline; padding:5px 0;
    border-bottom:1px solid rgba(28,32,40,.7);
  }
  .bubble.bot li.ann:last-child{border-bottom:none;}
  .ann-d{
    flex:0 0 auto; font-size:11px; color:var(--txt-3);
    font-variant-numeric:tabular-nums; letter-spacing:.2px;
    padding-top:1px;
  }
  .ann-t{flex:1; color:#d5dae3; line-height:1.62;}
  .bubble.bot hr{border:none; border-top:1px solid var(--line-2); margin:15px 0;}
  .bubble.bot code{
    background:#05070a; border:1px solid var(--line-2);
    padding:1.5px 5px; border-radius:4px;
    font-family:"JetBrains Mono","Consolas",monospace; font-size:12px; color:#8fc7ff;
  }
  .tbl-wrap{overflow-x:auto; margin:11px 0;}
  .bubble.bot table{
    border-collapse:collapse; font-size:12.8px; width:100%;
    font-variant-numeric:tabular-nums;
  }
  .bubble.bot th{
    background:var(--panel-2); color:var(--txt-2); font-weight:600;
    text-align:left; padding:8px 13px; font-size:11.5px;
    letter-spacing:.3px; border-bottom:1px solid var(--line-2);
  }
  .bubble.bot td{
    padding:8px 13px; border-bottom:1px solid var(--line);
    color:#d5dae3;
  }
  .bubble.bot tr:last-child td{border-bottom:none;}

  details.raw{
    margin-top:14px; border-top:1px solid var(--line);
    padding-top:11px;
  }
  details.raw summary{
    cursor:pointer; font-size:11.5px; color:var(--txt-3);
    list-style:none; user-select:none; display:flex; align-items:center; gap:6px;
  }
  details.raw summary::-webkit-details-marker{display:none;}
  details.raw summary::before{
    content:"▸"; font-size:9px; transition:.16s; display:inline-block;
  }
  details.raw[open] summary::before{transform:rotate(90deg);}
  details.raw summary:hover{color:var(--txt-2);}
  details.raw pre{
    margin-top:10px; padding:13px 15px; border-radius:9px;
    background:#05070a; border:1px solid var(--line-2);
    font-family:"JetBrains Mono","Consolas",monospace;
    font-size:11.5px; line-height:1.72; color:#8b98a8;
    overflow-x:auto; white-space:pre;
  }

  .meta{
    display:flex; gap:16px; flex-wrap:wrap;
    margin-top:10px; padding-left:3px;
    font-size:11px; color:var(--txt-3);
  }
  .meta b{color:var(--txt-3); font-weight:500;}
  .meta .ok{color:var(--green);}
  .meta .fail{color:var(--red);}

  /* 加载 */
  .loading{display:flex; align-items:center; gap:11px; padding:3px 0 3px 31px;}
  .dots{display:flex; gap:4px;}
  .dots i{
    width:5px; height:5px; border-radius:50%; background:var(--accent);
    animation:blink 1.25s infinite;
  }
  .dots i:nth-child(2){animation-delay:.18s;}
  .dots i:nth-child(3){animation-delay:.36s;}
  @keyframes blink{0%,60%,100%{opacity:.22; transform:scale(.85);} 30%{opacity:1; transform:scale(1);}}
  .load-txt{font-size:12.5px; color:var(--txt-3);}
  .load-t{font-size:11px; color:var(--txt-3); font-variant-numeric:tabular-nums; margin-left:auto; padding-right:4px;}

  /* 输入区 */
  .composer{padding:16px 32px 22px; border-top:1px solid var(--line); background:var(--bg);}
  .composer-inner{max-width:820px; margin:0 auto;}
  .box{
    display:flex; align-items:flex-end; gap:10px;
    background:var(--panel); border:1px solid var(--line-2);
    border-radius:14px; padding:9px 9px 9px 16px;
    transition:.18s;
  }
  .box:focus-within{
    border-color:rgba(91,140,255,.5);
    box-shadow:0 0 0 3px rgba(91,140,255,.09);
  }
  textarea{
    flex:1; background:none; border:none; outline:none; resize:none;
    color:var(--txt); font-size:14px; line-height:1.6;
    font-family:inherit; padding:7px 0; max-height:150px; min-height:24px;
  }
  textarea::placeholder{color:var(--txt-3);}
  #send{
    flex:0 0 auto; width:36px; height:36px; border-radius:10px;
    background:linear-gradient(135deg,#5b8cff,#4a7ef0);
    border:none; color:#fff; cursor:pointer;
    display:flex; align-items:center; justify-content:center;
    transition:.16s; font-size:15px;
  }
  #send:hover:not(:disabled){filter:brightness(1.12); transform:translateY(-1px);}
  #send:disabled{opacity:.35; cursor:not-allowed; transform:none;}
  .hint{
    margin-top:9px; font-size:11px; color:var(--txt-3);
    display:flex; justify-content:space-between; padding:0 3px;
  }
  .hint kbd{
    background:var(--panel-2); border:1px solid var(--line-2);
    border-radius:4px; padding:1px 5px; font-size:10px;
    font-family:inherit; color:var(--txt-2);
  }

  ::-webkit-scrollbar{width:9px; height:9px;}
  ::-webkit-scrollbar-track{background:transparent;}
  ::-webkit-scrollbar-thumb{background:#1e2430; border-radius:5px; border:2px solid var(--bg);}
  ::-webkit-scrollbar-thumb:hover{background:#2b3441;}

  @media (max-width:900px){
    .sidebar{display:none;}
    .chat-inner,.composer-inner{padding-left:18px; padding-right:18px;}
    .cards{grid-template-columns:1fr;}
    .welcome h2{font-size:22px;}
  }</style>
</head>
<body>
<div class="app">

  <aside class="sidebar">
    <div class="brand">
      <div class="logo">M</div>
      <div class="brand-txt">
        <h1>MarketAgent</h1>
        <p>A2A 多智能体投研</p>
      </div>
    </div>

    <div class="side-sec">
      <div class="side-title">数据规模</div>
      <div class="stat"><span class="stat-k">上市公司公告</span><span class="stat-v" id="s-ann">—</span></div>
      <div class="stat"><span class="stat-k">上市公司</span><span class="stat-v" id="s-cmp">—</span></div>
      <div class="stat"><span class="stat-k">向量索引</span><span class="stat-v" id="s-vec">—</span></div>
      <div class="stat"><span class="stat-k">检索方式</span><span class="stat-v dim">召回 50 → 精排 10</span></div>
    </div>

    <div class="side-sec">
      <div class="side-title">Agent</div>
      <div class="agent">
        <span class="dot" id="d-announcement"></span>
        <span class="agent-name">公告检索</span>
        <span class="agent-port">:5005</span>
      </div>
      <div class="agent">
        <span class="dot" id="d-company"></span>
        <span class="agent-name">公司画像</span>
        <span class="agent-port">:5006</span>
      </div>
      <div class="agent">
        <span class="dot" id="d-market"></span>
        <span class="agent-name">市场数据</span>
        <span class="agent-port">:5007</span>
      </div>
    </div>

    <div class="side-sec">
      <div class="side-title">试试这样问</div>
      <div class="ex" data-q="最近有哪些公司回购股份"><span class="ex-ico">›</span><span>最近有哪些公司回购股份</span></div>
      <div class="ex" data-q="浦发银行最近有什么公告"><span class="ex-ico">›</span><span>浦发银行最近有什么公告</span></div>
      <div class="ex" data-q="贵州燃气发了多少公告"><span class="ex-ico">›</span><span>贵州燃气发了多少公告</span></div>
      <div class="ex" data-q="今天大盘怎么样，浦发银行是什么公司"><span class="ex-ico">›</span><span>今天大盘怎么样，浦发银行是什么公司</span></div>
      <div class="ex" data-q="人民币兑美元汇率"><span class="ex-ico">›</span><span>人民币兑美元汇率</span></div>
    </div>
  </aside>

  <div class="main">
    <div class="chat" id="chat">
      <div class="chat-inner" id="chatInner">
        <div class="welcome" id="welcome">
          <div class="welcome-badge">A2A PROTOCOL v1.0 · 3 AGENTS</div>
          <h2>上市公司公告情报助手</h2>
          <p>三个独立部署的 agent 通过 A2A 协议协作<br>
             每条答案都可溯源到公告 ID · 股票代码 · 数据来源</p>

          <div class="cards">
            <div class="card">
              <span class="card-ico">◈</span>
              <h3>公告语义检索</h3>
              <p>两阶段检索，覆盖 13 万条公告</p>
            </div>
            <div class="card">
              <span class="card-ico">◉</span>
              <h3>公司画像</h3>
              <p>基本信息、行业归属与公告统计</p>
            </div>
            <div class="card">
              <span class="card-ico">◐</span>
              <h3>市场数据</h3>
              <p>指数行情、人民币汇率、黄金</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="composer">
      <div class="composer-inner">
        <div class="box">
          <textarea id="q" rows="1" placeholder="输入问题，例如：浦发银行最近有什么公告"></textarea>
          <button id="send" title="发送">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2.2"
                 stroke-linecap="round" stroke-linejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"></line>
              <polyline points="5 12 12 5 19 12"></polyline>
            </svg>
          </button>
        </div>
        <div class="hint">
          <span><kbd>Enter</kbd> 发送 · <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行</span>
          <span id="hint-right"></span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
const chat      = document.getElementById('chat');
const chatInner = document.getElementById('chatInner');
const welcome   = document.getElementById('welcome');
const input     = document.getElementById('q');
const btn       = document.getElementById('send');

const AGENTS = {
  announcement: {label:'公告检索', ico:'◈'},
  company:      {label:'公司画像', ico:'◉'},
  market:       {label:'市场数据', ico:'◐'}
};

/* ---------- 侧栏：示例问题 ---------- */
document.querySelectorAll('.ex').forEach(el => {
  el.onclick = () => { input.value = el.dataset.q; ask(); };
});

/* ---------- 输入框 ---------- */
function autoGrow(){
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 150) + 'px';
}
input.addEventListener('input', autoGrow);
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); }
});
btn.onclick = ask;

/* ---------- 侧栏：数据规模 + agent 状态 ---------- */
function fmt(n){ return n ? n.toLocaleString('en-US') : '—'; }

fetch('/api/stats').then(r => r.json()).then(d => {
  document.getElementById('s-ann').textContent = fmt(d.announcements);
  document.getElementById('s-cmp').textContent = fmt(d.companies);
  document.getElementById('s-vec').textContent = fmt(d.vectors);
}).catch(() => {});

fetch('/api/status').then(r => r.json()).then(d => {
  const online = d.agents || [];
  Object.keys(AGENTS).forEach(k => {
    const dot = document.getElementById('d-' + k);
    if (!dot) return;
    dot.className = 'dot ' + (online.includes(k) ? 'on' : 'off');
  });
  document.getElementById('hint-right').textContent =
    online.length + '/3 agent 在线';
}).catch(() => {
  document.getElementById('hint-right').textContent = 'agent 离线';
});

/* ---------- 工具 ---------- */
function esc(s){
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* 轻量 markdown：标题 / 表格 / 列表 / 加粗 / 行内代码 / 分隔线 */
function md(src){
  // 【结构化数据】之后的 JSON 折叠展示，避免占满屏幕
  let main = src, extra = '';
  const MARK = '【结构化数据】';
  const i = src.indexOf(MARK);
  if (i >= 0){
    main  = src.slice(0, i);
    extra = src.slice(i + MARK.length).trim();
  }

  let h = renderMd(main);
  if (extra){
    h += '<details class="raw"><summary>查看结构化数据</summary>' +
         '<pre>' + esc(extra) + '</pre></details>';
  }
  return h;
}

function renderMd(src){
  const lines = esc(src).split('\n');
  const out = [];
  let tbl = [], para = [];

  const flushPara = () => {
    if (!para.length) return;
    out.push('<p>' + para.join('<br>') + '</p>');
    para = [];
  };
  const flushTbl = () => {
    if (!tbl.length) return;
    const rows = tbl.filter(r => !r.every(c => /^:?-{2,}:?$/.test(c)));
    if (rows.length){
      let h = '<div class="tbl-wrap"><table>';
      rows.forEach((r, i) => {
        const tag = i === 0 ? 'th' : 'td';
        h += '<tr>' + r.map(c => '<' + tag + '>' + c + '</' + tag + '>').join('') + '</tr>';
      });
      out.push(h + '</table></div>');
    }
    tbl = [];
  };

  for (const raw of lines){
    const t = raw.trim();

    // 缩进的 "[2026-10-30] 公告标题" 行 → 带日期标签的列表项
    const dt = t.match(/^\[(\d{4}-\d{2}-\d{2})\]\s*(.+)$/);
    if (dt && /^\s{2,}/.test(raw)){
      flushPara(); flushTbl();
      out.push('<li class="ann"><span class="ann-d">' + dt[1] + '</span>' +
               '<span class="ann-t">' + dt[2] + '</span></li>');
      continue;
    }

    if (t.startsWith('|') && t.endsWith('|') && t.length > 2){
      flushPara();
      tbl.push(t.slice(1, -1).split('|').map(c => c.trim()));
      continue;
    }
    flushTbl();

    if (t === ''){ flushPara(); continue; }
    if (/^-{3,}$/.test(t)){ flushPara(); out.push('<hr>'); continue; }
    // 【xxx】 整行 → 标题
    if (/^【[^】]{1,40}】$/.test(t)){
      flushPara();
      out.push('<h3 class="ttl">' + t + '</h3>');
      continue;
    }
    if (/^#{1,6}\s/.test(t)){
      flushPara();
      const lv = Math.min(t.match(/^#+/)[0].length + 2, 6);
      out.push('<h' + lv + '>' + t.replace(/^#+\s*/, '') + '</h' + lv + '>');
      continue;
    }
    if (/^[-*]\s+/.test(t)){
      flushPara();
      out.push('<li>' + t.replace(/^[-*]\s+/, '') + '</li>');
      continue;
    }
    if (/^\d+[.、]\s*/.test(t)){
      flushPara();
      out.push('<li>' + t.replace(/^\d+[.、]\s*/, '') + '</li>');
      continue;
    }
    para.push(t);
  }
  flushPara(); flushTbl();

  let h = out.join('\n');
  h = h.replace(/(?:<li[^>]*>[\s\S]*?<\/li>\n?)+/g, m => '<ul>' + m + '</ul>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  return h;
}

function scrollDown(){ chat.scrollTop = chat.scrollHeight; }

/* ---------- 主流程 ---------- */
async function ask(){
  const q = input.value.trim();
  if (!q || btn.disabled) return;

  if (welcome) welcome.style.display = 'none';

  btn.disabled = true;
  input.value = '';
  autoGrow();

  // 用户消息
  chatInner.insertAdjacentHTML('beforeend',
    '<div class="msg user"><div class="bubble">' + esc(q) + '</div></div>');

  // 加载占位
  const lid = 'l' + Date.now();
  chatInner.insertAdjacentHTML('beforeend',
    '<div class="msg" id="' + lid + '">' +
      '<div class="loading">' +
        '<div class="dots"><i></i><i></i><i></i></div>' +
        '<span class="load-txt">正在识别意图…</span>' +
        '<span class="load-t">0.0s</span>' +
      '</div>' +
    '</div>');
  scrollDown();

  const t0 = Date.now();
  const tick = setInterval(() => {
    const el = document.querySelector('#' + lid + ' .load-t');
    if (el) el.textContent = ((Date.now() - t0) / 1000).toFixed(1) + 's';
  }, 100);
  // 分阶段提示（纯文案，让等待不那么枯燥）
  const stage1 = setTimeout(() => {
    const el = document.querySelector('#' + lid + ' .load-txt');
    if (el) el.textContent = '正在调度 agent…';
  }, 1600);
  const stage2 = setTimeout(() => {
    const el = document.querySelector('#' + lid + ' .load-txt');
    if (el) el.textContent = '正在汇总结果…';
  }, 5000);

  try {
    const r = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q})
    });
    const d = await r.json();
    // 优先用后端返回的真实耗时（前端 Date.now 在无头/虚拟时钟下不可靠）
    const dt = (d.elapsed != null)
      ? Number(d.elapsed).toFixed(1)
      : ((Date.now() - t0) / 1000).toFixed(1);

    document.getElementById(lid)?.remove();

    if (d.error){
      chatInner.insertAdjacentHTML('beforeend',
        '<div class="msg">' +
          '<div class="bot-head">' +
            '<div class="bot-ava">M</div>' +
            '<span class="bot-name">出错了</span>' +
          '</div>' +
          '<div class="bubble bot">' + esc(d.error) + '</div>' +
        '</div>');
    } else {
      const intents = d.intents || [];
      const called  = Object.keys(d.results || {});

      const chips = intents.map(k => {
        const a = AGENTS[k] || {label:k, ico:'·'};
        return '<span class="chip">' + a.ico + ' ' + esc(a.label) + '</span>';
      }).join('');

      const meta = called.length
        ? called.map(k => {
            const a = AGENTS[k] || {label:k};
            const ok = (d.results[k] || {}).ok;
            return '<span><b>' + esc(a.label) + '</b> ' +
                   '<span class="' + (ok ? 'ok' : 'fail') + '">' +
                   (ok ? '成功' : '失败') + '</span></span>';
          }).join('')
        : '<span>未调用 agent</span>';

      chatInner.insertAdjacentHTML('beforeend',
        '<div class="msg">' +
          '<div class="bot-head">' +
            '<div class="bot-ava">M</div>' +
            '<span class="bot-name">MarketAgent</span>' +
            '<div class="bot-chips">' + chips + '</div>' +
          '</div>' +
          '<div class="bubble bot">' + md(d.answer || '') + '</div>' +
          '<div class="meta">' +
            '<span><b>耗时</b> ' + dt + 's</span>' +
            meta +
          '</div>' +
        '</div>');
    }
  } catch (e) {
    document.getElementById(lid)?.remove();
    chatInner.insertAdjacentHTML('beforeend',
      '<div class="msg"><div class="bubble bot">请求失败：' + esc(e.message) + '</div></div>');
  } finally {
    clearInterval(tick);
    clearTimeout(stage1);
    clearTimeout(stage2);
    btn.disabled = false;
    input.focus();
    scrollDown();
  }
}

input.focus();

/* ---------- 支持 ?q=xxx 直接提问（便于分享链接 / 自动化截图） ---------- */
(function(){
  const q = new URLSearchParams(location.search).get('q');
  if (q && q.trim()){
    input.value = q.trim();
    setTimeout(ask, 150);
  }
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/api/stats")
async def stats() -> JSONResponse:
    """数据规模统计（缓存 5 分钟）。"""
    now = time.time()
    if _stats_cache["data"] and now - _stats_cache["at"] < _STATS_TTL:
        return JSONResponse(_stats_cache["data"])

    data = await asyncio.to_thread(_collect_stats)
    _stats_cache["data"] = data
    _stats_cache["at"] = now
    return JSONResponse(data)


@app.get("/api/status")
async def status() -> JSONResponse:
    """检查 agent 在线状态。"""
    cards = await asyncio.to_thread(_orch.discover_agents)
    return JSONResponse({"count": len(cards), "agents": list(cards)})


@app.post("/api/ask")
async def ask(request: Request) -> JSONResponse:
    """核心问答接口。"""
    try:
        body = await request.json()
        query = (body.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "query 不能为空"}, status_code=400)

        t0 = time.time()
        # ⚠️ FastAPI 已在事件循环中，必须用 arun() 而不是 run()
        # run() 内部会调 asyncio.run()，会报 "cannot be called from a running event loop"
        result = await _orch.arun(query)
        elapsed = time.time() - t0

        logger.info(
            "Web 查询完成 query=%r 意图=%s 耗时=%.1fs",
            query[:40],
            result.get("intents"),
            elapsed,
        )

        return JSONResponse(
            {
                "answer": result.get("answer", ""),
                "intents": result.get("intents", []),
                "results": result.get("results", {}),
                "elapsed": round(elapsed, 2),
            }
        )

    except Exception as e:  # noqa: BLE001
        log_exception(logger, e, "Web 查询失败")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


def main() -> None:
    import uvicorn

    logger.info("=" * 68)
    logger.info("启动 Web 界面")
    logger.info("  地址: http://127.0.0.1:8085")
    logger.info("=" * 68)

    uvicorn.run(app, host="0.0.0.0", port=8085, log_level="warning")


if __name__ == "__main__":
    main()
