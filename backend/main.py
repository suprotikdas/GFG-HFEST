# ═══════════════════════════════════════════════════════════════════════════
# AI Sales Data Analyst — FastAPI Backend v4.0
# Features: Supabase DB + Rate Limit + Query Cache + CSV→DB import
# Run locally:  uvicorn backend.main:app --reload --port 8000
# ═══════════════════════════════════════════════════════════════════════════

import re, json, io, base64, warnings, time, os, hashlib, threading
from collections import deque
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from dotenv import load_dotenv

import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "gemini-2.0-flash"
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
PORT         = int(os.getenv("PORT", "8000"))
_BASE        = Path(__file__).parent.parent
DATA_DIR     = _BASE / "data"
FRONT_DIR    = _BASE / "frontend"

# ── Chart colors ──────────────────────────────────────────────────────────────
PALETTE = ["#00F3FF","#BC13FE","#FF6B35","#00FF9D","#FFD600","#FF3860","#1F77B4","#8C564B"]
BG=  "#060810"; CARD= "#0F1520"; CARD2= "#141B26"; BORDER= "#1E2535"
TEXT="#E2E8F0"; MUTED="#4A5568"; GRID= "#0D1520"

# ── Global state ──────────────────────────────────────────────────────────────
df_global  = None
rag_chunks = []
filename   = ""

# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMITER — controls Gemini API calls
# Free tier: 15 req/min, 1500 req/day
# ═══════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self):
        self.minute_calls = deque()
        self.day_calls    = deque()
        self.lock         = threading.Lock()
        self.MAX_MIN      = 12
        self.MAX_DAY      = 1400

    def _clean(self):
        now = time.time()
        while self.minute_calls and now - self.minute_calls[0] > 60:    self.minute_calls.popleft()
        while self.day_calls    and now - self.day_calls[0]    > 86400: self.day_calls.popleft()

    def can_call(self):
        with self.lock:
            self._clean()
            now = time.time()
            if len(self.minute_calls) >= self.MAX_MIN:
                wait = int(60 - (now - self.minute_calls[0])) + 1
                return False, "minute", wait
            if len(self.day_calls) >= self.MAX_DAY:
                wait = int(86400 - (now - self.day_calls[0])) + 1
                return False, "day", wait
            return True, "ok", 0

    def record(self):
        with self.lock:
            now = time.time()
            self.minute_calls.append(now)
            self.day_calls.append(now)

    def status(self):
        with self.lock:
            self._clean()
            return {
                "calls_this_minute": len(self.minute_calls),
                "calls_today":       len(self.day_calls),
                "max_per_minute":    self.MAX_MIN,
                "max_per_day":       self.MAX_DAY,
                "minute_remaining":  self.MAX_MIN - len(self.minute_calls),
                "day_remaining":     self.MAX_DAY - len(self.day_calls),
            }

rate_limiter = RateLimiter()

# ═══════════════════════════════════════════════════════════════════════════
# QUERY CACHE — instant results for repeated queries
# ═══════════════════════════════════════════════════════════════════════════
_query_cache = {}
_CACHE_MAX   = 100

def _cache_key(q): return hashlib.md5(q.lower().strip().encode()).hexdigest()
def _cache_get(q): return _query_cache.get(_cache_key(q))
def _cache_set(q, result):
    _query_cache[_cache_key(q)] = result
    if len(_query_cache) > _CACHE_MAX:
        del _query_cache[next(iter(_query_cache))]
def _cache_clear(): _query_cache.clear()

# ═══════════════════════════════════════════════════════════════════════════
# SUPABASE / POSTGRESQL
# ═══════════════════════════════════════════════════════════════════════════
_db_conn = None

def get_db():
    global _db_conn
    if not DATABASE_URL: return None
    try:
        import psycopg2
        if _db_conn is None or _db_conn.closed:
            _db_conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            _db_conn.autocommit = True
        return _db_conn
    except Exception as e:
        print(f"[db] Connect error: {e}"); _db_conn = None; return None

def run_sql_db(sql):
    import psycopg2.extras
    conn = get_db()
    if not conn: raise Exception("No DB connection")
    pg = re.sub(r"strftime\s*\((\w+)\s*,\s*'%Y-%m'\s*\)", r"TO_CHAR(\1,'YYYY-MM')", sql, flags=re.IGNORECASE)
    pg = re.sub(r"strftime\s*\((\w+)\s*,\s*'%Y'\s*\)",    r"TO_CHAR(\1,'YYYY')",    pg,  flags=re.IGNORECASE)
    pg = re.sub(r"strftime\s*\((\w+)\s*,\s*'%m'\s*\)",    r"TO_CHAR(\1,'MM')",      pg,  flags=re.IGNORECASE)
    pg = re.sub(r"year\s*\(\s*(\w+)\s*\)", r"EXTRACT(YEAR FROM \1)::int", pg, flags=re.IGNORECASE)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(pg)
            rows = cur.fetchall()
            return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception as e:
        global _db_conn; _db_conn = None; raise e

def save_csv_to_db(df, table="sales_data"):
    conn = get_db()
    if not conn: return False
    try:
        import psycopg2.extras
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
            cols = ["id SERIAL PRIMARY KEY"]
            for c in df.columns:
                dtype = str(df[c].dtype)
                if "date" in c.lower() or "datetime" in dtype: pg = "DATE"
                elif "int" in dtype:   pg = "INTEGER"
                elif "float" in dtype: pg = "NUMERIC"
                else:                  pg = "TEXT"
                cols.append(f'"{c}" {pg}')
            cur.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
            rows = []
            for _, row in df.iterrows():
                vals = []
                for v in row:
                    if pd.isna(v): vals.append(None)
                    elif hasattr(v,'item'): vals.append(v.item())
                    else: vals.append(str(v)[:10] if hasattr(v,'date') else v)
                rows.append(tuple(vals))
            col_names = ', '.join(f'"{c}"' for c in df.columns)
            psycopg2.extras.execute_values(cur, f"INSERT INTO {table} ({col_names}) VALUES %s", rows, page_size=1000)
        _cache_clear()
        print(f"[db] ✅ Saved {len(df):,} rows to {table}")
        return True
    except Exception as e:
        print(f"[db] Save error: {e}"); return False

def init_db():
    if not DATABASE_URL: print("[db] No DATABASE_URL — using pandas/DuckDB"); return False
    try:
        conn = get_db()
        if not conn: return False
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sales_data")
            count = cur.fetchone()[0]
            print(f"[db] ✅ Supabase — {count:,} rows in sales_data")
            return True
    except Exception as e:
        print(f"[db] Table not found: {e}"); return False

# ═══════════════════════════════════════════════════════════════════════════
# SQL EXECUTOR — Supabase → DuckDB → Pandas fallback
# ═══════════════════════════════════════════════════════════════════════════
def run_sql(df_src, sql):
    if DATABASE_URL:
        try: return run_sql_db(sql)
        except Exception as e: print(f"[db] Falling back: {e}")
    try:
        import duckdb
        df = df_src.copy() if df_src is not None else pd.DataFrame()
        return duckdb.query(sql.replace("sales_data","df")).df()
    except Exception: pass
    return _pandas_fallback(df_src, sql)

def _pandas_fallback(df_src, sql):
    df = df_src.copy() if df_src is not None else pd.DataFrame()
    if df.empty: return df
    for col in df.columns:
        if "date" in col.lower(): df[col] = pd.to_datetime(df[col], errors="coerce")
    def _clause(p,t):
        m = re.search(p, t, re.IGNORECASE|re.DOTALL)
        return m.group(1).strip() if m else None
    sel   = _clause(r"SELECT\s+(.+?)\s+FROM\b", sql) or "*"
    where = _clause(r"\bWHERE\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", sql)
    grp   = _clause(r"\bGROUP\s+BY\s+(.+?)(?:\bORDER\b|\bLIMIT\b|\bHAVING\b|$)", sql)
    order = _clause(r"\bORDER\s+BY\s+(.+?)(?:\bLIMIT\b|$)", sql)
    lim   = _clause(r"\bLIMIT\s+(\d+)", sql)
    if where:
        m = re.search(r"strftime\s*\(\s*(\w+)\s*,\s*'(%[YmdHMS%-]+)'\s*\)\s*=\s*'([^']+)'", where, re.IGNORECASE)
        if m:
            dc,fmt,val = m.group(1),m.group(2),m.group(3)
            a = next((c for c in df.columns if c.lower()==dc.lower()), None)
            if a: df = df[df[a].dt.strftime(fmt)==val]
        m2 = re.search(r"year\s*\(\s*(\w+)\s*\)\s*=\s*(\d{4})", where, re.IGNORECASE)
        if m2:
            dc,yr = m2.group(1),int(m2.group(2))
            a = next((c for c in df.columns if c.lower()==dc.lower()), None)
            if a: df = df[df[a].dt.year==yr]
    AGG = {"SUM":"sum","COUNT":"count","AVG":"mean","MIN":"min","MAX":"max"}
    items = []
    def _split(clause):
        parts,depth,cur = [],0,""
        for ch in clause:
            if ch=="(": depth+=1
            elif ch==")": depth-=1
            if ch=="," and depth==0: parts.append(cur.strip()); cur=""
            else: cur+=ch
        if cur.strip(): parts.append(cur.strip())
        return parts
    for part in _split(sel):
        m = re.match(r"strftime\s*\(\s*(\w+)\s*,\s*'([^']+)'\s*\)\s*(?:AS\s+(\w+))?", part, re.IGNORECASE)
        if m:
            src,fmt,alias = m.group(1),m.group(2),m.group(3) or m.group(1)
            a = next((c for c in df.columns if c.lower()==src.lower()), None)
            if a: df[alias] = df[a].dt.strftime(fmt)
            items.append((alias,alias,None,True)); continue
        m = re.match(r"(SUM|COUNT|AVG|MIN|MAX)\s*\(\s*\*?(\w+)?\s*\)\s*(?:AS\s+(\w+))?", part, re.IGNORECASE)
        if m:
            func,src,alias = m.group(1).upper(),m.group(2) or df.columns[0],m.group(3) or f"{m.group(1).lower()}_{m.group(2) or df.columns[0]}"
            items.append((alias,src,AGG[func],False)); continue
        am = re.search(r"\bAS\s+(\w+)", part, re.IGNORECASE)
        alias = am.group(1) if am else None
        col_raw = re.sub(r"\s+AS\s+\w+","",part,flags=re.IGNORECASE).strip().strip('"').strip("'")
        items.append((alias or col_raw, col_raw, None, False))
    if grp:
        gcols = [g.strip().strip('"').strip("'") for g in grp.split(",")]
        gcols = [g for g in gcols if g in df.columns]
        agg_s = {src:(alias,func) for alias,src,func,_ in items if func and src in df.columns}
        if gcols and agg_s:
            try:
                grouped = df.groupby(gcols).agg({s:f for s,(a,f) in agg_s.items()}).reset_index()
                df = grouped.rename(columns={s:a for s,(a,_) in agg_s.items()})
            except Exception: pass
        elif gcols:
            keep = [c for c in gcols if c in df.columns]
            if keep: df = df[keep].drop_duplicates()
    else:
        if sel.strip()!="*":
            km = {src:alias for alias,src,func,_ in items if func is None and src in df.columns}
            if km: df = df[list(km.keys())].rename(columns=km)
    if order:
        bys,ascs = [],[]
        for p in order.split(","):
            p = p.strip(); asc = not p.upper().endswith("DESC")
            cn = re.sub(r"\s+(ASC|DESC)$","",p,flags=re.IGNORECASE).strip().strip('"').strip("'")
            if cn in df.columns: bys.append(cn); ascs.append(asc)
        if bys: df = df.sort_values(bys, ascending=ascs)
    if lim: df = df.head(int(lim))
    return df.reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATASET STATS
# ═══════════════════════════════════════════════════════════════════════════
def compute_dataset_stats(df):
    s = {"total_rows":len(df),"total_columns":len(df.columns),"columns":df.columns.tolist()}
    dc = next((c for c in df.columns if "date" in c.lower()), None)
    if dc:
        try:
            dates = pd.to_datetime(df[dc], errors="coerce").dropna()
            s.update({"date_min":str(dates.min())[:10],"date_max":str(dates.max())[:10],
                      "date_min_m":str(dates.min())[:7],"date_max_m":str(dates.max())[:7],
                      "total_months":len(dates.dt.to_period("M").unique()),
                      "total_years":sorted(dates.dt.year.unique().tolist())})
        except Exception: pass
    if "total_revenue" in df.columns:
        rev = pd.to_numeric(df["total_revenue"], errors="coerce").dropna()
        s.update({"total_revenue":float(rev.sum()),"avg_revenue":float(rev.mean())})
    if dc and "total_revenue" in df.columns:
        try:
            df2 = df.copy(); df2[dc] = pd.to_datetime(df2[dc], errors="coerce")
            yearly = df2.groupby(df2[dc].dt.year)["total_revenue"].sum()
            yrs = sorted(yearly.index.tolist())
            if len(yrs)>=2:
                y1,y2 = float(yearly[yrs[-2]]),float(yearly[yrs[-1]])
                s.update({"growth_pct":round(((y2-y1)/y1)*100,2) if y1 else 0,
                           "growth_y1":yrs[-2],"growth_y2":yrs[-1],"revenue_y1":round(y1,2),"revenue_y2":round(y2,2)})
            monthly = df2.groupby(df2[dc].dt.to_period("M"))["total_revenue"].sum()
            if len(monthly)>=6:
                f3,l3 = float(monthly.head(3).mean()),float(monthly.tail(3).mean())
                s.update({"monthly_growth_pct":round(((l3-f3)/f3)*100,2) if f3 else 0,
                           "best_month":str(monthly.idxmax()),"best_month_rev":round(float(monthly.max()),2)})
        except Exception as e: print(f"Stats err: {e}")
    if "product_category" in df.columns and "total_revenue" in df.columns:
        try:
            cats = df.groupby("product_category")["total_revenue"].sum().sort_values(ascending=False)
            s["categories"] = {k:round(float(v),2) for k,v in cats.items()}
        except Exception: pass
    if "customer_region" in df.columns and "total_revenue" in df.columns:
        try:
            regs = df.groupby("customer_region")["total_revenue"].sum().sort_values(ascending=False)
            s["regions"] = {k:round(float(v),2) for k,v in regs.items()}
        except Exception: pass
    if "quantity_sold" in df.columns:
        try: s["total_quantity"] = int(df["quantity_sold"].sum())
        except Exception: pass
    if "rating" in df.columns:
        try: s["avg_rating"] = round(float(df["rating"].mean()),2)
        except Exception: pass
    return s

# ═══════════════════════════════════════════════════════════════════════════
# RAG ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def build_rag_chunks(df, fname="dataset"):
    chunks = [f"Dataset '{fname}': {len(df)} rows, {len(df.columns)} cols: {', '.join(df.columns.tolist())}."]
    for col in df.columns:
        chunks.append(f"Column '{col}': type={df[col].dtype}, {df[col].nunique()} unique, sample: {df[col].dropna().head(3).tolist()}.")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols: chunks.append(f"Numeric summary:\n{df[num_cols].describe().to_string()}")
    for col in df.select_dtypes(include=["object","string"]).columns[:6]:
        chunks.append(f"Top values in '{col}': {json.dumps(df[col].value_counts().head(10).to_dict())}")
    for dc in [c for c in df.columns if "date" in c.lower()]:
        try: chunks.append(f"'{dc}' spans {str(df[dc].min())[:10]} to {str(df[dc].max())[:10]}.")
        except Exception: pass
    return chunks

def retrieve_context(chunks, query, top_k=4):
    words = set(query.lower().split())
    scored = sorted([(len(words & set(c.lower().split())), c) for c in chunks], key=lambda x:-x[0])
    return "\n".join(c for _,c in scored[:top_k])

def get_schema_text(df):
    lines = ["Table: sales_data","Columns:"]
    for col in df.columns:
        lines.append(f"  - {col} ({df[col].dtype}) — {df[col].nunique()} unique, sample: {df[col].dropna().head(3).tolist()}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# GEMINI ENGINE
# ═══════════════════════════════════════════════════════════════════════════
SQL_PROMPT = """You are an expert SQL analyst.
DATASET CONTEXT:\n{context}\nFULL SCHEMA:\n{schema}\nUSER QUESTION:\n{question}
RULES:
1. Generate ONLY a single SELECT query.
2. Table name is ALWAYS: sales_data
3. Date ops: TO_CHAR(order_date,'YYYY-MM') for month, TO_CHAR(order_date,'YYYY') for year
4. Top N: ORDER BY ... DESC LIMIT N
5. Columns: order_date,product_id,product_category,price,discount_percent,quantity_sold,customer_region,payment_method,rating,review_count,discounted_price,total_revenue
6. Return ONLY the SQL — no markdown, no explanation.
SQL QUERY:"""

CHART_PROMPT = """Pick best chart for this SQL result.
SQL: {sql}\nColumns: {columns}\nRows: {row_count}
Return ONLY JSON: chart_type(line/bar/horizontal_bar/pie/donut/area/scatter), x_col, y_col, title, reason
JSON:"""

CHAT_SYSTEM = """You are a helpful data analyst for an Amazon Sales dataset.
Facts: 50,000 rows, 2022-2023, $32.87M revenue, +0.54% YoY growth.
Columns: order_date,product_id,product_category,price,discount_percent,quantity_sold,customer_region,payment_method,rating,review_count,discounted_price,total_revenue
Categories: Books,Fashion,Sports,Beauty,Electronics,Home & Kitchen
Regions: North America,Asia,Europe,Middle East — Top: Middle East
Payments: UPI,Credit Card,Wallet,Cash on Delivery,Debit Card
Be concise, friendly, data-driven. Use bullets for lists."""

_BLOCKED = re.compile(r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|EXEC)\b", re.IGNORECASE)

def _call_gemini(prompt, retries=2):
    can, reason, wait = rate_limiter.can_call()
    if not can:
        raise Exception(f"RATE_LIMIT_{reason.upper()}:{wait}")
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(retries):
        try:
            rate_limiter.record()
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err:
                if attempt < retries-1: time.sleep(65)
                else: raise Exception("RATE_LIMIT_MINUTE:65")
            elif "leaked" in err.lower() or "403" in err:
                raise Exception("API_KEY_LEAKED")
            else: raise e

def generate_sql(question, df, chunks):
    ctx    = retrieve_context(chunks, question)
    schema = get_schema_text(df) if df is not None else "Table: sales_data"
    raw    = _call_gemini(SQL_PROMPT.format(context=ctx, schema=schema, question=question))
    raw    = re.sub(r"```sql|```","",raw,flags=re.IGNORECASE)
    m      = re.search(r"(SELECT\b.*)", raw, re.IGNORECASE|re.DOTALL)
    sql    = m.group(1).strip() if m else raw.strip()
    if _BLOCKED.search(sql) or not sql.upper().startswith("SELECT"):
        raise ValueError(f"Unsafe SQL: {sql[:80]}")
    return sql

def choose_chart(sql, columns, row_count):
    try:
        raw  = re.sub(r"```json|```","",_call_gemini(CHART_PROMPT.format(sql=sql,columns=columns,row_count=row_count))).strip()
        jm   = re.search(r"\{.*\}", raw, re.DOTALL)
        meta = json.loads(jm.group(0) if jm else raw)
        if meta.get("x_col") not in columns: meta["x_col"] = columns[0]
        if meta.get("y_col") not in columns:
            nc = [c for c in columns if c != meta.get("x_col")]
            meta["y_col"] = nc[-1] if nc else columns[-1]
        return meta
    except Exception:
        cl = " ".join(columns).lower()
        if any(k in cl for k in ["month","year","date","time"]):
            return {"chart_type":"area","x_col":columns[0],"y_col":columns[-1],"title":"Trend","reason":"Time series."}
        if row_count<=6:
            return {"chart_type":"donut","x_col":columns[0],"y_col":columns[-1],"title":"Distribution","reason":"Few categories."}
        return {"chart_type":"bar","x_col":columns[0],"y_col":columns[-1],"title":"Comparison","reason":"Bar chart."}

# ═══════════════════════════════════════════════════════════════════════════
# CHART ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def _fmt(v):
    try:
        v=float(v)
        if abs(v)>=1e6: return f"{v/1e6:.1f}M"
        if abs(v)>=1e3: return f"{v/1e3:.0f}K"
        return f"{v:,.1f}" if v!=int(v) else f"{v:,.0f}"
    except: return str(v)

def _fig(w=12,h=5):
    fig,ax=plt.subplots(figsize=(w,h)); fig.patch.set_facecolor(BG); ax.set_facecolor(CARD); return fig,ax

def _style(ax,title,xl="",yl=""):
    ax.set_title(title,color=TEXT,fontsize=13,fontweight="bold",pad=14,loc="left")
    ax.tick_params(colors=MUTED,labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.set_axisbelow(True); ax.yaxis.grid(True,color=GRID,linewidth=0.5,linestyle="--"); ax.xaxis.grid(False)
    if xl: ax.set_xlabel(xl,color=MUTED,fontsize=9)
    if yl: ax.set_ylabel(yl,color=MUTED,fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: _fmt(x)))

def _b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor=BG); buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def _res(df,meta):
    cols=df.columns.tolist(); xc=meta.get("x_col"); yc=meta.get("y_col")
    if not xc or xc not in cols: xc=cols[0]
    if not yc or yc not in cols:
        nc=df.select_dtypes(include="number").columns.tolist(); yc=next((c for c in nc if c!=xc),cols[-1])
    return xc,yc

def chart_bar(df,meta):
    xc,yc=_res(df,meta); title=meta.get("title","Result")
    if len(df)>20: df=df.head(20); title+=" (Top 20)"
    xv=df[xc].astype(str).tolist(); yv=pd.to_numeric(df[yc],errors="coerce").fillna(0).tolist()
    fig,ax=_fig(max(10,len(xv)*0.8),5)
    colors=[PALETTE[i%len(PALETTE)] for i in range(len(xv))]
    bars=ax.bar(range(len(xv)),yv,color=colors,edgecolor=BG,linewidth=0.5,width=0.65,zorder=2)
    my=max(yv) if yv else 1
    for bar,val in zip(bars,yv):
        if val>0: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+my*0.015,_fmt(val),ha="center",va="bottom",fontsize=8,color=TEXT,fontweight="600")
    ax.set_xticks(range(len(xv))); ax.set_xticklabels(xv,rotation=30,ha="right",fontsize=9,color=TEXT); ax.set_xlim(-0.6,len(xv)-0.4)
    _style(ax,title,xc.replace("_"," ").title(),yc.replace("_"," ").title()); fig.tight_layout(pad=1.6); return _b64(fig)

def chart_hbar(df,meta):
    xc,yc=_res(df,meta); title=meta.get("title","Result")
    if len(df)>20: df=df.head(20)
    xv=list(reversed(df[xc].astype(str).tolist())); yv=list(reversed(pd.to_numeric(df[yc],errors="coerce").fillna(0).tolist()))
    colors=list(reversed([PALETTE[i%len(PALETTE)] for i in range(len(xv))]))
    fig,ax=_fig(11,max(4,len(xv)*0.5))
    bars=ax.barh(range(len(xv)),yv,color=colors,edgecolor=BG,linewidth=0.5,height=0.6,zorder=2)
    my=max(yv) if yv else 1
    for bar,val in zip(bars,yv): ax.text(bar.get_width()+my*0.01,bar.get_y()+bar.get_height()/2,_fmt(val),va="center",ha="left",fontsize=8.5,color=TEXT)
    ax.set_yticks(range(len(xv))); ax.set_yticklabels(xv,fontsize=9,color=TEXT)
    ax.set_facecolor(CARD); ax.set_title(title,color=TEXT,fontsize=13,fontweight="bold",pad=14,loc="left")
    ax.tick_params(colors=MUTED,labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:_fmt(x))); ax.xaxis.grid(True,color=GRID,linewidth=0.5,linestyle="--"); ax.yaxis.grid(False); ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6); return _b64(fig)

def chart_line(df,meta):
    xc,yc=_res(df,meta); xv=df[xc].astype(str).tolist(); yv=pd.to_numeric(df[yc],errors="coerce").fillna(0)
    fig,ax=_fig(13,5)
    ax.plot(range(len(xv)),yv,color=PALETTE[0],linewidth=2.5,marker="o",markersize=5,markerfacecolor=BG,markeredgecolor=PALETTE[0],markeredgewidth=2,zorder=3)
    ax.fill_between(range(len(xv)),yv,alpha=0.10,color=PALETTE[0])
    if len(yv)>0:
        pi=int(np.argmax(yv))
        ax.annotate(f"Peak\n{_fmt(yv.iloc[pi])}",xy=(pi,yv.iloc[pi]),xytext=(0,18),textcoords="offset points",ha="center",fontsize=8,color=PALETTE[0],fontweight="bold",arrowprops=dict(arrowstyle="->",color=PALETTE[0],lw=1.2))
    step=max(1,len(xv)//12); ax.set_xticks(range(0,len(xv),step)); ax.set_xticklabels(xv[::step],rotation=40,ha="right",fontsize=8.5,color=TEXT)
    _style(ax,meta.get("title","Trend"),xc.replace("_"," ").title(),yc.replace("_"," ").title()); fig.tight_layout(pad=1.6); return _b64(fig)

def chart_area(df,meta):
    xc,yc=_res(df,meta); xv=df[xc].astype(str).tolist(); yv=pd.to_numeric(df[yc],errors="coerce").fillna(0)
    fig,ax=_fig(13,5)
    ax.fill_between(range(len(xv)),yv,alpha=0.22,color=PALETTE[0]); ax.plot(range(len(xv)),yv,color=PALETTE[0],linewidth=2.5,zorder=3)
    if len(yv)>=4:
        roll=pd.Series(yv).rolling(3,center=True).mean()
        ax.plot(range(len(xv)),roll,color=PALETTE[1],linewidth=1.8,linestyle="--",alpha=0.8,label="3-pt avg")
        ax.legend(facecolor=CARD2,edgecolor=BORDER,labelcolor=TEXT,fontsize=8,loc="upper left")
    step=max(1,len(xv)//12); ax.set_xticks(range(0,len(xv),step)); ax.set_xticklabels(xv[::step],rotation=40,ha="right",fontsize=8.5,color=TEXT)
    _style(ax,meta.get("title","Trend"),xc.replace("_"," ").title(),yc.replace("_"," ").title()); fig.tight_layout(pad=1.6); return _b64(fig)

def chart_pie(df,meta):
    xc,yc=_res(df,meta)
    if len(df)>8: df=df.head(8)
    lbs=df[xc].astype(str).tolist(); vals=pd.to_numeric(df[yc],errors="coerce").fillna(0).tolist()
    fig,ax=_fig(9,6)
    w,_,at=ax.pie(vals,labels=None,colors=PALETTE[:len(lbs)],autopct=lambda p:f"{p:.1f}%" if p>3 else "",startangle=90,wedgeprops={"edgecolor":BG,"linewidth":2},textprops={"color":TEXT,"fontsize":9})
    for a in at: a.set_fontsize(8.5); a.set_fontweight("bold"); a.set_color(TEXT)
    ax.legend(w,[f"{l}  ({_fmt(v)})" for l,v in zip(lbs,vals)],loc="lower center",bbox_to_anchor=(0.5,-0.15),ncol=min(3,len(lbs)),facecolor=CARD2,edgecolor=BORDER,labelcolor=TEXT,fontsize=8.5)
    ax.set_title(meta.get("title","Distribution"),color=TEXT,fontsize=13,fontweight="bold",pad=14); fig.tight_layout(pad=1.6); return _b64(fig)

def chart_donut(df,meta):
    xc,yc=_res(df,meta)
    if len(df)>8: df=df.head(8)
    lbs=df[xc].astype(str).tolist(); vals=pd.to_numeric(df[yc],errors="coerce").fillna(0).tolist()
    fig,ax=_fig(9,6)
    w,_,at=ax.pie(vals,labels=None,colors=PALETTE[:len(lbs)],autopct=lambda p:f"{p:.1f}%" if p>3 else "",startangle=90,wedgeprops={"edgecolor":BG,"linewidth":2,"width":0.52},textprops={"color":TEXT,"fontsize":9})
    for a in at: a.set_fontsize(8.5); a.set_fontweight("bold"); a.set_color(TEXT)
    ax.text(0,0,f"{_fmt(sum(vals))}\nTotal",ha="center",va="center",fontsize=12,color=TEXT,fontweight="bold",linespacing=1.6)
    ax.legend(w,[f"{l}  ({_fmt(v)})" for l,v in zip(lbs,vals)],loc="lower center",bbox_to_anchor=(0.5,-0.15),ncol=min(3,len(lbs)),facecolor=CARD2,edgecolor=BORDER,labelcolor=TEXT,fontsize=8.5)
    ax.set_title(meta.get("title","Distribution"),color=TEXT,fontsize=13,fontweight="bold",pad=14); fig.tight_layout(pad=1.6); return _b64(fig)

def chart_scatter(df,meta):
    xc,yc=_res(df,meta); xv=df[xc].astype(str).tolist(); yv=pd.to_numeric(df[yc],errors="coerce").fillna(0).tolist()
    colors=[PALETTE[i%len(PALETTE)] for i in range(len(xv))]
    fig,ax=_fig(12,5)
    ax.scatter(range(len(xv)),yv,color=colors,s=90,alpha=0.85,edgecolors=BG,linewidth=0.8,zorder=3)
    step=max(1,len(xv)//12); ax.set_xticks(range(0,len(xv),step)); ax.set_xticklabels(xv[::step],rotation=35,ha="right",fontsize=9,color=TEXT)
    _style(ax,meta.get("title","Scatter"),xc.replace("_"," ").title(),yc.replace("_"," ").title()); fig.tight_layout(pad=1.6); return _b64(fig)

def render_chart(df,meta):
    dispatch={"bar":chart_bar,"horizontal_bar":chart_hbar,"line":chart_line,"area":chart_area,"pie":chart_pie,"donut":chart_donut,"scatter":chart_scatter}
    return dispatch.get(meta.get("chart_type","bar").lower().strip(), chart_bar)(df.copy(), meta)

# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="AI Sales Data Analyst", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False)

@app.middleware("http")
async def cors_fix(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(status_code=200, headers={"Access-Control-Allow-Origin":"*","Access-Control-Allow-Methods":"GET, POST, OPTIONS","Access-Control-Allow-Headers":"*"})
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

class QueryRequest(BaseModel): question: str
class ChatRequest(BaseModel): message: str; history: list = []

@app.get("/")
def root(): return FileResponse(str(FRONT_DIR / "index.html"))

@app.get("/health")
def health():
    return {"status":"ok","dataset_loaded":df_global is not None,"filename":filename,
            "rows":len(df_global) if df_global is not None else 0,
            "revenue":float(df_global["total_revenue"].sum()) if df_global is not None and "total_revenue" in df_global.columns else 0,
            "model":MODEL_NAME,"database":"supabase" if DATABASE_URL else "pandas","rate_limit":rate_limiter.status()}

@app.get("/stats")
def get_stats():
    if df_global is None: raise HTTPException(400,"No dataset loaded.")
    return compute_dataset_stats(df_global)

@app.get("/rate-limit")
def get_rate_limit(): return rate_limiter.status()

@app.get("/schema")
def schema():
    if df_global is None: raise HTTPException(400,"No dataset loaded.")
    return {"filename":filename,"rows":len(df_global),"columns":df_global.columns.tolist(),"stats":compute_dataset_stats(df_global)}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global df_global, rag_chunks, filename
    if not file.filename.endswith(".csv"): raise HTTPException(400,"Only CSV files supported.")
    raw = await file.read()
    try: df_global = pd.read_csv(io.BytesIO(raw))
    except Exception: df_global = pd.read_csv(io.BytesIO(raw), encoding="latin1")
    for col in df_global.columns:
        if "date" in col.lower(): df_global[col] = pd.to_datetime(df_global[col], errors="coerce")
    filename = file.filename; rag_chunks = build_rag_chunks(df_global, filename)
    db_saved = save_csv_to_db(df_global, "sales_data") if DATABASE_URL else False
    return {"success":True,"filename":filename,"rows":len(df_global),"columns":df_global.columns.tolist(),"db_saved":db_saved,"stats":compute_dataset_stats(df_global)}

@app.post("/query")
def query(req: QueryRequest):
    # Cache hit — instant
    cached = _cache_get(req.question)
    if cached: return cached

    if df_global is None and not DATABASE_URL:
        return {"error":True,"code":"NO_DATA","message":"No dataset loaded.","hint":"Restart the server — it auto-loads Amazon_Sales.csv on startup."}
    if not GEMINI_KEY:
        return {"error":True,"code":"NO_KEY","message":"Gemini API key not set.","hint":"Set GEMINI_API_KEY in your .env file and restart."}

    # Rate limit check
    can, reason, wait = rate_limiter.can_call()
    if not can:
        return {"error":True,"code":f"RATE_LIMIT_{reason.upper()}","message":f"Rate limit reached — wait {wait}s.",
                "hint":f"⏳ Wait {wait} seconds then try again. Your query: '{req.question}'","wait_seconds":wait}

    # Generate SQL
    sql = None
    try:
        sql = generate_sql(req.question, df_global, rag_chunks)
    except Exception as e:
        err = str(e)
        if "RATE_LIMIT" in err:
            wait = int(err.split(":")[1]) if ":" in err else 60
            return {"error":True,"code":"RATE_LIMIT","message":f"Rate limit — wait {wait}s.","hint":f"⏳ Wait {wait} seconds.","wait_seconds":wait}
        if "API_KEY_LEAKED" in err:
            return {"error":True,"code":"API_KEY_LEAKED","message":"API key was flagged as leaked.","hint":"Get a new key at https://aistudio.google.com/apikey and update it in Render Environment."}
        return {"error":True,"code":"SQL_GEN_FAILED","message":f"Could not generate SQL: {err[:200]}","hint":"Try rephrasing. E.g. 'Show total revenue by product category'"}

    # Execute SQL
    try:
        result_df = run_sql(df_global, sql)
    except Exception as e:
        return {"error":True,"code":"SQL_EXEC_FAILED","message":f"Query failed: {str(e)[:200]}","hint":"Try a simpler question.","sql":sql}

    if result_df is None or result_df.empty:
        return {"sql":sql,"rows":0,"columns":[],"data":[],"chart_b64":None,"chart_meta":None,"stats":None}

    # Choose chart + render
    chart_meta = choose_chart(sql, result_df.columns.tolist(), len(result_df))
    chart_b64  = None
    try: chart_b64 = render_chart(result_df, chart_meta)
    except Exception as e: print(f"Chart error: {e}")

    # Stats
    yc = chart_meta.get("y_col", result_df.columns[-1])
    xc = chart_meta.get("x_col", result_df.columns[0])
    yv = pd.to_numeric(result_df[yc], errors="coerce").dropna()
    stats = None
    if len(yv) > 0:
        mi,ni = yv.idxmax(),yv.idxmin()
        stats = {"total":float(yv.sum()),"average":float(yv.mean()),"max_val":float(yv.max()),"min_val":float(yv.min()),
                 "max_label":str(result_df.loc[mi,xc]) if xc in result_df.columns else "","min_label":str(result_df.loc[ni,xc]) if xc in result_df.columns else ""}

    # Serialize
    data = []
    for row in result_df.head(200).to_dict(orient="records"):
        clean = {}
        for k,v in row.items():
            try:
                if pd.isna(v): clean[k]=None; continue
            except Exception: pass
            clean[k] = v.item() if hasattr(v,"item") else (str(v)[:10] if hasattr(v,"date") else v)
        data.append(clean)

    result = {"sql":sql,"rows":len(result_df),"columns":result_df.columns.tolist(),"data":data,"chart_b64":chart_b64,"chart_meta":chart_meta,"stats":stats}
    _cache_set(req.question, result)
    return result

@app.post("/chat")
def chat(req: ChatRequest):
    if not GEMINI_KEY: raise HTTPException(400,"GEMINI_API_KEY not set.")
    can, reason, wait = rate_limiter.can_call()
    if not can:
        return {"reply":f"⏳ Rate limit — wait {wait} seconds and try again.","rate_limited":True,"wait_seconds":wait}
    conv = [CHAT_SYSTEM, "\n\nCONVERSATION:"]
    for msg in req.history[-10:]:
        conv.append(f"\n{'User' if msg.get('role')=='user' else 'Assistant'}: {msg.get('content','')}")
    conv.append(f"\nUser: {req.message}\nAssistant:")
    try:
        reply = _call_gemini("\n".join(conv))
        return {"reply":reply,"status":"ok"}
    except Exception as e:
        err = str(e)
        if "RATE_LIMIT" in err:
            wait = int(err.split(":")[1]) if ":" in err else 60
            return {"reply":f"⏳ Rate limit — wait {wait}s.","rate_limited":True,"wait_seconds":wait}
        raise HTTPException(500, f"AI error: {err}")

@app.on_event("startup")
def startup():
    global df_global, rag_chunks, filename
    db_ok = init_db()
    csv = DATA_DIR / "Amazon_Sales.csv"
    if csv.exists():
        try:
            df_global = pd.read_csv(str(csv))
            for col in df_global.columns:
                if "date" in col.lower(): df_global[col] = pd.to_datetime(df_global[col], errors="coerce")
            filename   = "Amazon_Sales.csv"
            rag_chunks = build_rag_chunks(df_global, filename)
            s = compute_dataset_stats(df_global)
            print(f"[startup] ✅ Loaded {len(df_global):,} rows")
            print(f"[startup]    Revenue:  ${s.get('total_revenue',0):,.2f}")
            print(f"[startup]    Growth:   {s.get('growth_pct',0):+.2f}%")
            print(f"[startup]    Database: {'Supabase ✅' if db_ok else 'pandas fallback'}")
            print(f"[startup]    Rate limit: {rate_limiter.MAX_MIN}/min · {rate_limiter.MAX_DAY}/day")
        except Exception as e:
            print(f"[startup] ❌ CSV load failed: {e}")
