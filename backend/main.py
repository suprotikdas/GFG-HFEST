# ═══════════════════════════════════════════════════════════════════════════
# AI Sales Data Analyst — FastAPI Backend v4.0
# Features: Supabase DB + Rate Limit Control + Query Cache + CSV→DB import
# ═══════════════════════════════════════════════════════════════════════════

import re, json, io, base64, warnings, time, os, hashlib, threading
from collections import deque
from datetime import datetime, timedelta
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
PORT         = int(os.getenv("PORT", "8080"))
_BASE        = Path(__file__).parent.parent
DATA_DIR     = _BASE / "data"
FRONT_DIR    = _BASE / "frontend"

# ── Chart colors ──────────────────────────────────────────────────────────────
PALETTE = ["#00F3FF","#BC13FE","#FF6B35","#00FF9D","#FFD600","#FF3860","#1F77B4","#8C564B"]
BG      = "#060810"
CARD    = "#0F1520"
CARD2   = "#141B26"
BORDER  = "#1E2535"
TEXT    = "#E2E8F0"
MUTED   = "#4A5568"
ACCENT  = "#00F3FF"
GRID    = "#0D1520"

# ── Global state ──────────────────────────────────────────────────────────────
df_global  = None
rag_chunks = []
filename   = ""

# ═══════════════════════════════════════════════════════════════════════════
# RATE LIMIT CONTROLLER
# Controls Gemini API calls to stay within free tier limits
# Free tier: 15 requests/minute, 1500 requests/day
# ═══════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self):
        self.minute_calls  = deque()   # timestamps of calls in last 60s
        self.day_calls     = deque()   # timestamps of calls in last 24h
        self.lock          = threading.Lock()
        self.MAX_PER_MIN   = 12        # keep buffer below 15
        self.MAX_PER_DAY   = 1400      # keep buffer below 1500
        self.queue         = deque()   # pending requests
        self.queue_lock    = threading.Lock()

    def _clean_old(self):
        now = time.time()
        while self.minute_calls and now - self.minute_calls[0] > 60:
            self.minute_calls.popleft()
        while self.day_calls and now - self.day_calls[0] > 86400:
            self.day_calls.popleft()

    def can_call(self) -> tuple[bool, str, int]:
        """Returns (can_call, reason, wait_seconds)"""
        with self.lock:
            self._clean_old()
            now = time.time()
            if len(self.minute_calls) >= self.MAX_PER_MIN:
                wait = int(60 - (now - self.minute_calls[0])) + 1
                return False, "minute_limit", wait
            if len(self.day_calls) >= self.MAX_PER_DAY:
                wait = int(86400 - (now - self.day_calls[0])) + 1
                return False, "day_limit", wait
            return True, "ok", 0

    def record_call(self):
        with self.lock:
            now = time.time()
            self.minute_calls.append(now)
            self.day_calls.append(now)

    def status(self) -> dict:
        with self.lock:
            self._clean_old()
            return {
                "calls_this_minute": len(self.minute_calls),
                "calls_today":       len(self.day_calls),
                "max_per_minute":    self.MAX_PER_MIN,
                "max_per_day":       self.MAX_PER_DAY,
                "minute_remaining":  self.MAX_PER_MIN - len(self.minute_calls),
                "day_remaining":     self.MAX_PER_DAY - len(self.day_calls),
            }

rate_limiter = RateLimiter()

# ═══════════════════════════════════════════════════════════════════════════
# QUERY CACHE — instant results for repeated queries
# ═══════════════════════════════════════════════════════════════════════════
_query_cache = {}
_CACHE_MAX   = 100

def _cache_key(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def _cache_get(question: str):
    return _query_cache.get(_cache_key(question))

def _cache_set(question: str, result: dict):
    key = _cache_key(question)
    _query_cache[key] = result
    if len(_query_cache) > _CACHE_MAX:
        oldest = next(iter(_query_cache))
        del _query_cache[oldest]

def _cache_clear():
    _query_cache.clear()

# ═══════════════════════════════════════════════════════════════════════════
# SUPABASE / POSTGRESQL
# ═══════════════════════════════════════════════════════════════════════════
_db_conn = None

def get_db():
    global _db_conn
    if not DATABASE_URL:
        return None
    try:
        import psycopg2
        import psycopg2.extras
        if _db_conn is None or _db_conn.closed:
            _db_conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            _db_conn.autocommit = True
        return _db_conn
    except Exception as e:
        print(f"[db] Connect error: {e}")
        _db_conn = None
        return None


def run_sql_db(sql: str) -> pd.DataFrame:
    """Run SQL on Supabase PostgreSQL."""
    import psycopg2.extras
    conn = get_db()
    if conn is None:
        raise Exception("No DB connection")
    # Convert DuckDB syntax → PostgreSQL
    pg = sql
    pg = re.sub(r"strftime\s*\((\w+)\s*,\s*'%Y-%m'\s*\)",
                r"TO_CHAR(\1, 'YYYY-MM')", pg, flags=re.IGNORECASE)
    pg = re.sub(r"strftime\s*\((\w+)\s*,\s*'%Y'\s*\)",
                r"TO_CHAR(\1, 'YYYY')", pg, flags=re.IGNORECASE)
    pg = re.sub(r"strftime\s*\((\w+)\s*,\s*'%m'\s*\)",
                r"TO_CHAR(\1, 'MM')", pg, flags=re.IGNORECASE)
    # Fix date comparison
    pg = re.sub(r"year\s*\(\s*(\w+)\s*\)", r"EXTRACT(YEAR FROM \1)", pg, flags=re.IGNORECASE)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(pg)
            rows = cur.fetchall()
            return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    except Exception as e:
        global _db_conn
        _db_conn = None
        raise e


def save_csv_to_db(df: pd.DataFrame, table_name: str = "sales_data"):
    """Save DataFrame to Supabase — replaces existing data."""
    conn = get_db()
    if conn is None:
        print("[db] No connection — skipping DB save")
        return False
    try:
        import psycopg2.extras
        with conn.cursor() as cur:
            # Drop and recreate table
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            # Build CREATE TABLE from DataFrame dtypes
            col_defs = ["id SERIAL PRIMARY KEY"]
            for col in df.columns:
                dtype = str(df[col].dtype)
                if "date" in col.lower() or "datetime" in dtype:
                    pg_type = "DATE"
                elif "int" in dtype:
                    pg_type = "INTEGER"
                elif "float" in dtype or "numeric" in dtype:
                    pg_type = "NUMERIC"
                else:
                    pg_type = "TEXT"
                col_defs.append(f'"{col}" {pg_type}')
            cur.execute(f"CREATE TABLE {table_name} ({', '.join(col_defs)})")
            # Bulk insert
            cols = [f'"{c}"' for c in df.columns]
            args = []
            for _, row in df.iterrows():
                vals = []
                for v in row:
                    if pd.isna(v):
                        vals.append(None)
                    elif hasattr(v, 'item'):
                        vals.append(v.item())
                    else:
                        vals.append(str(v)[:10] if hasattr(v, 'date') else v)
                args.append(tuple(vals))
            insert_sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES %s"
            psycopg2.extras.execute_values(cur, insert_sql, args, page_size=1000)
        print(f"[db] ✅ Saved {len(df):,} rows to {table_name}")
        _cache_clear()
        return True
    except Exception as e:
        print(f"[db] Save error: {e}")
        return False


def init_db():
    if not DATABASE_URL:
        print("[db] No DATABASE_URL — using pandas/DuckDB")
        return False
    try:
        conn = get_db()
        if not conn:
            return False
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sales_data")
            count = cur.fetchone()[0]
            print(f"[db] ✅ Supabase ready — {count:,} rows in sales_data")
            return True
    except Exception as e:
        print(f"[db] Table not found or error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# SQL EXECUTOR — tries DB first, then DuckDB, then pandas
# ═══════════════════════════════════════════════════════════════════════════
def run_sql(df_src, sql):
    # 1. Try Supabase first
    if DATABASE_URL:
        try:
            result = run_sql_db(sql)
            return result
        except Exception as e:
            print(f"[db] Supabase failed, falling back: {e}")
    # 2. Try DuckDB
    try:
        import duckdb
        df = df_src.copy()
        return duckdb.query(sql.replace("sales_data", "df")).df()
    except Exception:
        pass
    # 3. Pandas fallback
    return _pandas_fallback(df_src, sql)


def _pandas_fallback(df_src, sql):
    df = df_src.copy() if df_src is not None else pd.DataFrame()
    if df.empty:
        return df
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    def _clause(pattern, text):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else None

    select_clause = _clause(r"SELECT\s+(.+?)\s+FROM\b", sql) or "*"
    where_clause  = _clause(r"\bWHERE\s+(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", sql)
    group_clause  = _clause(r"\bGROUP\s+BY\s+(.+?)(?:\bORDER\b|\bLIMIT\b|\bHAVING\b|$)", sql)
    order_clause  = _clause(r"\bORDER\s+BY\s+(.+?)(?:\bLIMIT\b|$)", sql)
    limit_clause  = _clause(r"\bLIMIT\s+(\d+)", sql)
    limit = int(limit_clause) if limit_clause else None

    if where_clause:
        m = re.search(r"strftime\s*\(\s*(\w+)\s*,\s*'(%[YmdHMS%-]+)'\s*\)\s*=\s*'([^']+)'",
                      where_clause, re.IGNORECASE)
        if m:
            dc, fmt, val = m.group(1), m.group(2), m.group(3)
            actual = next((c for c in df.columns if c.lower() == dc.lower()), None)
            if actual:
                df = df[df[actual].dt.strftime(fmt) == val]
        m2 = re.search(r"year\s*\(\s*(\w+)\s*\)\s*=\s*(\d{4})", where_clause, re.IGNORECASE)
        if m2:
            dc, yr = m2.group(1), int(m2.group(2))
            actual = next((c for c in df.columns if c.lower() == dc.lower()), None)
            if actual:
                df = df[df[actual].dt.year == yr]
        m3 = re.search(r"(\w+)\s*=\s*'([^']+)'", where_clause)
        if m3 and not m and not m2:
            dc, val = m3.group(1), m3.group(2)
            if dc in df.columns:
                df = df[df[dc] == val]

    AGG = {"SUM":"sum","COUNT":"count","AVG":"mean","MIN":"min","MAX":"max"}
    select_items = []

    def _split_select(clause):
        parts, depth, cur = [], 0, ""
        for ch in clause:
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur.strip()); cur = ""
            else:
                cur += ch
        if cur.strip(): parts.append(cur.strip())
        return parts

    for part in _split_select(select_clause):
        m = re.match(r"strftime\s*\(\s*(\w+)\s*,\s*'([^']+)'\s*\)\s*(?:AS\s+(\w+))?", part, re.IGNORECASE)
        if m:
            src, fmt, alias = m.group(1), m.group(2), m.group(3) or m.group(1)
            actual = next((c for c in df.columns if c.lower() == src.lower()), None)
            if actual:
                df[alias] = df[actual].dt.strftime(fmt)
            select_items.append((alias, alias, None, True)); continue
        m = re.match(r"(SUM|COUNT|AVG|MIN|MAX)\s*\(\s*\*?(\w+)?\s*\)\s*(?:AS\s+(\w+))?", part, re.IGNORECASE)
        if m:
            func  = m.group(1).upper()
            src   = m.group(2) or df.columns[0]
            alias = m.group(3) or f"{func.lower()}_{src}"
            select_items.append((alias, src, AGG[func], False)); continue
        alias_m = re.search(r"\bAS\s+(\w+)", part, re.IGNORECASE)
        alias   = alias_m.group(1) if alias_m else None
        col_raw = re.sub(r"\s+AS\s+\w+", "", part, flags=re.IGNORECASE).strip().strip('"').strip("'")
        alias   = alias or col_raw
        select_items.append((alias, col_raw, None, False))

    if group_clause:
        raw_gcols  = [g.strip().strip('"').strip("'") for g in group_clause.split(",")]
        group_cols = []
        for gc in raw_gcols:
            if gc in df.columns:
                group_cols.append(gc)
            else:
                found = next((alias for alias,_,_,_ in select_items if alias.lower()==gc.lower()), None)
                if found and found in df.columns:
                    group_cols.append(found)
        agg_specs = {}
        for alias, src, func, _ in select_items:
            if func and src in df.columns:
                agg_specs[src] = (alias, func)
        if group_cols and agg_specs:
            agg_dict = {src: func for src,(_, func) in agg_specs.items()}
            try:
                grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
                rename  = {src: alias for src,(alias,_) in agg_specs.items()}
                df      = grouped.rename(columns=rename)
            except Exception:
                pass
        elif group_cols:
            keep = [c for c in group_cols if c in df.columns]
            if keep:
                df = df[keep].drop_duplicates()
    else:
        if select_clause.strip() != "*":
            keep_map = {}
            for alias, src, func, _ in select_items:
                if func is None and src in df.columns:
                    keep_map[src] = alias
            if keep_map:
                df = df[list(keep_map.keys())].rename(columns=keep_map)

    if order_clause:
        by_cols, ascend = [], []
        for p in order_clause.split(","):
            p        = p.strip()
            asc      = not p.upper().endswith("DESC")
            col_name = re.sub(r"\s+(ASC|DESC)$", "", p, flags=re.IGNORECASE).strip().strip('"').strip("'")
            if col_name in df.columns:
                by_cols.append(col_name); ascend.append(asc)
        if by_cols:
            df = df.sort_values(by_cols, ascending=ascend)

    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATASET STATS
# ═══════════════════════════════════════════════════════════════════════════
def compute_dataset_stats(df: pd.DataFrame) -> dict:
    stats = {}
    stats["total_rows"]    = len(df)
    stats["total_columns"] = len(df.columns)
    stats["columns"]       = df.columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        dc = date_cols[0]
        try:
            dates = pd.to_datetime(df[dc], errors="coerce").dropna()
            stats["date_min"]     = str(dates.min())[:10]
            stats["date_max"]     = str(dates.max())[:10]
            stats["date_min_m"]   = str(dates.min())[:7]
            stats["date_max_m"]   = str(dates.max())[:7]
            stats["total_months"] = len(dates.dt.to_period("M").unique())
            stats["total_years"]  = sorted(dates.dt.year.unique().tolist())
        except Exception:
            pass
    if "total_revenue" in df.columns:
        rev = pd.to_numeric(df["total_revenue"], errors="coerce").dropna()
        stats["total_revenue"] = float(rev.sum())
        stats["avg_revenue"]   = float(rev.mean())
    if date_cols and "total_revenue" in df.columns:
        try:
            dc   = date_cols[0]
            df2  = df.copy()
            df2[dc] = pd.to_datetime(df2[dc], errors="coerce")
            df2["_year"] = df2[dc].dt.year
            yearly = df2.groupby("_year")["total_revenue"].sum()
            years  = sorted(yearly.index.tolist())
            if len(years) >= 2:
                y1 = float(yearly[years[-2]])
                y2 = float(yearly[years[-1]])
                stats["growth_pct"] = round(((y2-y1)/y1)*100, 2) if y1 else 0
                stats["growth_y1"]  = years[-2]
                stats["growth_y2"]  = years[-1]
                stats["revenue_y1"] = round(y1, 2)
                stats["revenue_y2"] = round(y2, 2)
            monthly = df2.groupby(df2[dc].dt.to_period("M"))["total_revenue"].sum()
            if len(monthly) >= 6:
                first3 = float(monthly.head(3).mean())
                last3  = float(monthly.tail(3).mean())
                stats["monthly_growth_pct"] = round(((last3-first3)/first3)*100, 2) if first3 else 0
                stats["best_month"]     = str(monthly.idxmax())
                stats["best_month_rev"] = round(float(monthly.max()), 2)
        except Exception as e:
            print(f"Growth calc error: {e}")
    if "product_category" in df.columns and "total_revenue" in df.columns:
        try:
            cats = df.groupby("product_category")["total_revenue"].sum().sort_values(ascending=False)
            stats["categories"] = {k: round(float(v),2) for k,v in cats.items()}
        except Exception:
            pass
    if "customer_region" in df.columns and "total_revenue" in df.columns:
        try:
            regs = df.groupby("customer_region")["total_revenue"].sum().sort_values(ascending=False)
            stats["regions"] = {k: round(float(v),2) for k,v in regs.items()}
        except Exception:
            pass
    if "quantity_sold" in df.columns:
        try:
            stats["total_quantity"] = int(df["quantity_sold"].sum())
        except Exception:
            pass
    if "rating" in df.columns:
        try:
            stats["avg_rating"] = round(float(df["rating"].mean()), 2)
        except Exception:
            pass
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# RAG ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def build_rag_chunks(df, fname="dataset"):
    chunks = []
    chunks.append(f"Dataset '{fname}': {len(df)} rows, {len(df.columns)} columns: {', '.join(df.columns.tolist())}.")
    for col in df.columns:
        dtype  = str(df[col].dtype)
        sample = df[col].dropna().head(3).tolist()
        unique = df[col].nunique()
        chunks.append(f"Column '{col}': type={dtype}, {unique} unique values, sample: {sample}.")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        chunks.append(f"Numeric summary:\n{df[num_cols].describe().to_string()}")
    cat_cols = df.select_dtypes(include=["object","string"]).columns.tolist()
    for col in cat_cols[:6]:
        vals = df[col].value_counts().head(10).to_dict()
        chunks.append(f"Top values in '{col}': {json.dumps(vals)}")
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for dc in date_cols:
        try:
            chunks.append(f"Column '{dc}' spans from {str(df[dc].min())[:10]} to {str(df[dc].max())[:10]}.")
        except Exception:
            pass
    return chunks


def retrieve_context(chunks, query, top_k=4):
    query_words = set(query.lower().split())
    scored = [(len(query_words & set(c.lower().split())), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(c for _, c in scored[:top_k])


def get_schema_text(df):
    lines = ["Table: sales_data", "Columns:"]
    for col in df.columns:
        lines.append(f"  - {col} ({df[col].dtype}) — {df[col].nunique()} unique, sample: {df[col].dropna().head(3).tolist()}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI ENGINE with rate limit control
# ═══════════════════════════════════════════════════════════════════════════
SQL_PROMPT = """You are an expert SQL analyst.

DATASET CONTEXT:
{context}

FULL SCHEMA:
{schema}

USER QUESTION:
{question}

RULES:
1. Generate ONLY a single SELECT query.
2. Table name is ALWAYS: sales_data
3. For date operations use standard SQL: TO_CHAR(order_date, 'YYYY-MM') for month
4. For top N: use ORDER BY ... DESC LIMIT N.
5. Exact column names: order_date, product_id, product_category, price, discount_percent,
   quantity_sold, customer_region, payment_method, rating, review_count, discounted_price, total_revenue
6. Return ONLY the SQL query — no explanation, no markdown fences.

SQL QUERY:"""

CHART_PROMPT = """Given this SQL and result columns, pick the best chart type.
SQL: {sql}
Result columns: {columns}
Row count: {row_count}
Return ONLY a JSON object: chart_type (line/bar/horizontal_bar/pie/donut/area/scatter), x_col, y_col, title, reason.
No markdown.
JSON:"""

CHAT_SYSTEM = """You are a helpful data analyst assistant for an Amazon Sales dataset.

DATASET FACTS:
- 50,000 rows, date range 2022-01-01 to 2023-12-31
- Total revenue: $32,866,573.74, YoY growth: +0.54%
- Columns: order_date, product_id, product_category, price, discount_percent,
  quantity_sold, customer_region, payment_method, rating, review_count, discounted_price, total_revenue
- Categories: Books, Fashion, Sports, Beauty, Electronics, Home & Kitchen
- Regions: North America, Asia, Europe, Middle East
- Payment: UPI, Credit Card, Wallet, Cash on Delivery, Debit Card
- Avg rating: ~3.0, Top region: Middle East, Top category: Beauty

Be concise, friendly and data-driven. Use bullet points for lists.
"""

_BLOCKED = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|EXEC)\b",
    re.IGNORECASE
)


def _call_gemini(prompt, retries=2):
    """Call Gemini with rate limit control."""
    # Check rate limit
    can, reason, wait = rate_limiter.can_call()
    if not can:
        if reason == "minute_limit":
            raise Exception(f"RATE_LIMIT_MINUTE:{wait}")
        else:
            raise Exception(f"RATE_LIMIT_DAY:{wait}")

    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(retries):
        try:
            rate_limiter.record_call()
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err:
                if attempt < retries - 1:
                    print("Rate limit from Gemini — waiting 65s...")
                    time.sleep(65)
                else:
                    raise Exception("RATE_LIMIT_MINUTE:65")
            elif "leaked" in err.lower() or "403" in err:
                raise Exception("API_KEY_LEAKED")
            else:
                raise e


def generate_sql(question, df, chunks):
    context = retrieve_context(chunks, question)
    schema  = get_schema_text(df) if df is not None else "Table: sales_data"
    prompt  = SQL_PROMPT.format(context=context, schema=schema, question=question)
    raw     = _call_gemini(prompt)
    raw     = re.sub(r"```sql|```", "", raw, flags=re.IGNORECASE)
    m       = re.search(r"(SELECT\b.*)", raw, re.IGNORECASE | re.DOTALL)
    sql     = m.group(1).strip() if m else raw.strip()
    if _BLOCKED.search(sql) or not sql.upper().startswith("SELECT"):
        raise ValueError(f"Unsafe SQL: {sql[:80]}")
    return sql


def choose_chart(sql, columns, row_count):
    try:
        prompt = CHART_PROMPT.format(sql=sql, columns=columns, row_count=row_count)
        raw    = re.sub(r"```json|```", "", _call_gemini(prompt)).strip()
        jm     = re.search(r"\{.*\}", raw, re.DOTALL)
        if jm: raw = jm.group(0)
        meta   = json.loads(raw)
        if meta.get("x_col") not in columns: meta["x_col"] = columns[0]
        if meta.get("y_col") not in columns:
            num_cols = [c for c in columns if c != meta.get("x_col")]
            meta["y_col"] = num_cols[-1] if num_cols else columns[-1]
        return meta
    except Exception:
        col_lower = " ".join(columns).lower()
        if any(k in col_lower for k in ["month","year","date","time"]):
            return {"chart_type":"area","x_col":columns[0],"y_col":columns[-1],"title":"Trend","reason":"Time series."}
        if row_count <= 6:
            return {"chart_type":"donut","x_col":columns[0],"y_col":columns[-1],"title":"Distribution","reason":"Few categories."}
        return {"chart_type":"bar","x_col":columns[0],"y_col":columns[-1],"title":"Comparison","reason":"Bar chart."}


# ═══════════════════════════════════════════════════════════════════════════
# CHART ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def _fmt(v):
    try:
        v = float(v)
        if abs(v) >= 1_000_000: return f"{v/1e6:.1f}M"
        if abs(v) >= 1_000:     return f"{v/1e3:.0f}K"
        return f"{v:,.1f}" if v != int(v) else f"{v:,.0f}"
    except Exception:
        return str(v)

def _base_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(CARD)
    return fig, ax

def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--")
    ax.xaxis.grid(False)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: _fmt(x)))

def _to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def _resolve(result_df, chart_meta):
    cols  = result_df.columns.tolist()
    x_col = chart_meta.get("x_col")
    y_col = chart_meta.get("y_col")
    if not x_col or x_col not in cols: x_col = cols[0]
    if not y_col or y_col not in cols:
        num_cols = result_df.select_dtypes(include="number").columns.tolist()
        y_col = next((c for c in num_cols if c != x_col), cols[-1])
    return x_col, y_col

def render_bar(df, meta):
    x_col, y_col = _resolve(df, meta)
    title = meta.get("title","Result")
    if len(df) > 20: df = df.head(20); title += " (Top 20)"
    xv = df[x_col].astype(str).tolist()
    yv = pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()
    fig, ax = _base_fig(max(10, len(xv)*0.8), 5)
    colors = [PALETTE[i%len(PALETTE)] for i in range(len(xv))]
    bars = ax.bar(range(len(xv)), yv, color=colors, edgecolor=BG, linewidth=0.5, width=0.65, zorder=2)
    max_y = max(yv) if yv else 1
    for bar, val in zip(bars, yv):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max_y*0.015,
                    _fmt(val), ha="center", va="bottom", fontsize=8, color=TEXT, fontweight="600")
    ax.set_xticks(range(len(xv))); ax.set_xticklabels(xv, rotation=30, ha="right", fontsize=9, color=TEXT)
    ax.set_xlim(-0.6, len(xv)-0.4)
    _style_ax(ax, title, x_col.replace("_"," ").title(), y_col.replace("_"," ").title())
    fig.tight_layout(pad=1.6); return _to_b64(fig)

def render_hbar(df, meta):
    x_col, y_col = _resolve(df, meta)
    title = meta.get("title","Result")
    if len(df) > 20: df = df.head(20)
    xv = list(reversed(df[x_col].astype(str).tolist()))
    yv = list(reversed(pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()))
    colors = [PALETTE[i%len(PALETTE)] for i in range(len(xv))]
    fig, ax = _base_fig(11, max(4, len(xv)*0.5))
    bars = ax.barh(range(len(xv)), yv, color=list(reversed(colors)), edgecolor=BG, linewidth=0.5, height=0.6, zorder=2)
    max_y = max(yv) if yv else 1
    for bar, val in zip(bars, yv):
        ax.text(bar.get_width()+max_y*0.01, bar.get_y()+bar.get_height()/2,
                _fmt(val), va="center", ha="left", fontsize=8.5, color=TEXT)
    ax.set_yticks(range(len(xv))); ax.set_yticklabels(xv, fontsize=9, color=TEXT)
    ax.set_facecolor(CARD); ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold", pad=14, loc="left")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor(BORDER)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: _fmt(x)))
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--"); ax.yaxis.grid(False); ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6); return _to_b64(fig)

def render_line(df, meta):
    x_col, y_col = _resolve(df, meta)
    title = meta.get("title","Trend")
    xv = df[x_col].astype(str).tolist()
    yv = pd.to_numeric(df[y_col], errors="coerce").fillna(0)
    fig, ax = _base_fig(13, 5)
    ax.plot(range(len(xv)), yv, color=PALETTE[0], linewidth=2.5, marker="o", markersize=5,
            markerfacecolor=BG, markeredgecolor=PALETTE[0], markeredgewidth=2, zorder=3)
    ax.fill_between(range(len(xv)), yv, alpha=0.10, color=PALETTE[0])
    if len(yv) > 0:
        pi = int(np.argmax(yv))
        ax.annotate(f"Peak\n{_fmt(yv.iloc[pi])}", xy=(pi, yv.iloc[pi]),
                    xytext=(0,18), textcoords="offset points", ha="center", fontsize=8,
                    color=PALETTE[0], fontweight="bold", arrowprops=dict(arrowstyle="->",color=PALETTE[0],lw=1.2))
    step = max(1, len(xv)//12)
    ax.set_xticks(range(0,len(xv),step)); ax.set_xticklabels(xv[::step], rotation=40, ha="right", fontsize=8.5, color=TEXT)
    _style_ax(ax, title, x_col.replace("_"," ").title(), y_col.replace("_"," ").title())
    fig.tight_layout(pad=1.6); return _to_b64(fig)

def render_area(df, meta):
    x_col, y_col = _resolve(df, meta)
    xv = df[x_col].astype(str).tolist()
    yv = pd.to_numeric(df[y_col], errors="coerce").fillna(0)
    fig, ax = _base_fig(13, 5)
    ax.fill_between(range(len(xv)), yv, alpha=0.22, color=PALETTE[0])
    ax.plot(range(len(xv)), yv, color=PALETTE[0], linewidth=2.5, zorder=3)
    if len(yv) >= 4:
        roll = pd.Series(yv).rolling(3, center=True).mean()
        ax.plot(range(len(xv)), roll, color=PALETTE[1], linewidth=1.8, linestyle="--", alpha=0.8, label="3-pt avg")
        ax.legend(facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, loc="upper left")
    step = max(1, len(xv)//12)
    ax.set_xticks(range(0,len(xv),step)); ax.set_xticklabels(xv[::step], rotation=40, ha="right", fontsize=8.5, color=TEXT)
    _style_ax(ax, meta.get("title","Trend"), x_col.replace("_"," ").title(), y_col.replace("_"," ").title())
    fig.tight_layout(pad=1.6); return _to_b64(fig)

def render_pie(df, meta):
    x_col, y_col = _resolve(df, meta)
    if len(df) > 8: df = df.head(8)
    labels = df[x_col].astype(str).tolist()
    values = pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()
    colors = PALETTE[:len(labels)]
    fig, ax = _base_fig(9, 6)
    wedges, _, autotexts = ax.pie(values, labels=None, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, wedgeprops={"edgecolor":BG,"linewidth":2},
        textprops={"color":TEXT,"fontsize":9})
    for at in autotexts: at.set_fontsize(8.5); at.set_fontweight("bold"); at.set_color(TEXT)
    legend_labels = [f"{l}  ({_fmt(v)})" for l,v in zip(labels,values)]
    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5,-0.15),
              ncol=min(3,len(labels)), facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
    ax.set_title(meta.get("title","Distribution"), color=TEXT, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout(pad=1.6); return _to_b64(fig)

def render_donut(df, meta):
    x_col, y_col = _resolve(df, meta)
    if len(df) > 8: df = df.head(8)
    labels = df[x_col].astype(str).tolist()
    values = pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()
    colors = PALETTE[:len(labels)]
    fig, ax = _base_fig(9, 6)
    wedges, _, autotexts = ax.pie(values, labels=None, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, wedgeprops={"edgecolor":BG,"linewidth":2,"width":0.52},
        textprops={"color":TEXT,"fontsize":9})
    for at in autotexts: at.set_fontsize(8.5); at.set_fontweight("bold"); at.set_color(TEXT)
    ax.text(0, 0, f"{_fmt(sum(values))}\nTotal", ha="center", va="center",
            fontsize=12, color=TEXT, fontweight="bold", linespacing=1.6)
    legend_labels = [f"{l}  ({_fmt(v)})" for l,v in zip(labels,values)]
    ax.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5,-0.15),
              ncol=min(3,len(labels)), facecolor=CARD2, edgecolor=BORDER, labelcolor=TEXT, fontsize=8.5)
    ax.set_title(meta.get("title","Distribution"), color=TEXT, fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout(pad=1.6); return _to_b64(fig)

def render_scatter(df, meta):
    x_col, y_col = _resolve(df, meta)
    xv = df[x_col].astype(str).tolist()
    yv = pd.to_numeric(df[y_col], errors="coerce").fillna(0).tolist()
    colors = [PALETTE[i%len(PALETTE)] for i in range(len(xv))]
    fig, ax = _base_fig(12, 5)
    ax.scatter(range(len(xv)), yv, color=colors, s=90, alpha=0.85, edgecolors=BG, linewidth=0.8, zorder=3)
    step = max(1, len(xv)//12)
    ax.set_xticks(range(0,len(xv),step)); ax.set_xticklabels(xv[::step], rotation=35, ha="right", fontsize=9, color=TEXT)
    _style_ax(ax, meta.get("title","Scatter"), x_col.replace("_"," ").title(), y_col.replace("_"," ").title())
    fig.tight_layout(pad=1.6); return _to_b64(fig)

def render_chart_to_b64(result_df, chart_meta):
    dispatch = {"bar":render_bar,"horizontal_bar":render_hbar,"line":render_line,
                "area":render_area,"pie":render_pie,"donut":render_donut,"scatter":render_scatter}
    ctype    = chart_meta.get("chart_type","bar").lower().strip()
    return dispatch.get(ctype, render_bar)(result_df.copy(), chart_meta)


# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="AI Sales Data Analyst", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

@app.middleware("http")
async def cors_fix(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(status_code=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        })
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


class QueryRequest(BaseModel):
    question: str

class ChatRequest(BaseModel):
    message: str
    history: list = []


@app.get("/")
def root():
    return FileResponse(str(FRONT_DIR / "index.html"))


@app.get("/health")
def health():
    rl = rate_limiter.status()
    return {
        "status":          "ok",
        "dataset_loaded":  df_global is not None,
        "filename":        filename,
        "rows":            len(df_global) if df_global is not None else 0,
        "revenue":         float(df_global["total_revenue"].sum()) if df_global is not None and "total_revenue" in df_global.columns else 0,
        "model":           MODEL_NAME,
        "database":        "supabase" if DATABASE_URL else "pandas",
        "rate_limit":      rl,
    }


@app.get("/stats")
def get_stats():
    if df_global is None:
        raise HTTPException(400, "No dataset loaded.")
    return compute_dataset_stats(df_global)


@app.get("/rate-limit")
def get_rate_limit():
    """Check current rate limit status."""
    return rate_limiter.status()


@app.get("/schema")
def schema():
    if df_global is None:
        raise HTTPException(400, "No dataset loaded.")
    return {
        "filename": filename,
        "rows":     len(df_global),
        "columns":  df_global.columns.tolist(),
        "schema":   get_schema_text(df_global),
        "stats":    compute_dataset_stats(df_global),
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global df_global, rag_chunks, filename
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported.")
    raw = await file.read()
    try:
        df_global = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df_global = pd.read_csv(io.BytesIO(raw), encoding="latin1")
    for col in df_global.columns:
        if "date" in col.lower():
            df_global[col] = pd.to_datetime(df_global[col], errors="coerce")
    filename   = file.filename
    rag_chunks = build_rag_chunks(df_global, filename)

    # Save to Supabase database
    db_saved = False
    if DATABASE_URL:
        print(f"[upload] Saving {len(df_global):,} rows to Supabase...")
        db_saved = save_csv_to_db(df_global, "sales_data")

    stats = compute_dataset_stats(df_global)
    return {
        "success":   True,
        "filename":  filename,
        "rows":      len(df_global),
        "columns":   df_global.columns.tolist(),
        "stats":     stats,
        "db_saved":  db_saved,
        "preview":   df_global.head(5).to_dict(orient="records"),
    }


@app.post("/query")
def query(req: QueryRequest):
    global df_global, rag_chunks

    # 0. Check cache — instant response
    cached = _cache_get(req.question)
    if cached:
        print(f"[cache] HIT: {req.question[:50]}")
        return cached

    # 1. Check data available
    if df_global is None and not DATABASE_URL:
        return {"error":True,"code":"NO_DATA","message":"No dataset loaded.",
                "hint":"Check terminal — server auto-loads Amazon_Sales.csv on startup."}
    if not GEMINI_KEY:
        return {"error":True,"code":"NO_KEY","message":"Gemini API key not set.",
                "hint":"Set GEMINI_API_KEY in Render environment variables."}

    # 2. Check rate limit BEFORE calling Gemini
    can, reason, wait = rate_limiter.can_call()
    if not can:
        if reason == "minute_limit":
            return {"error":True,"code":"RATE_LIMIT",
                    "message":f"Gemini rate limit reached — {wait} seconds until next available slot.",
                    "hint":f"⏳ Wait {wait} seconds and try again. Your query is: '{req.question}'",
                    "wait_seconds": wait}
        else:
            return {"error":True,"code":"RATE_LIMIT_DAY",
                    "message":"Daily Gemini quota reached.",
                    "hint":"You have used all 1400 requests for today. Resets at midnight UTC.",
                    "wait_seconds": wait}

    # 3. Generate SQL
    sql = None
    try:
        sql = generate_sql(req.question, df_global, rag_chunks)
    except Exception as e:
        err = str(e)
        if "RATE_LIMIT_MINUTE" in err:
            wait = int(err.split(":")[1]) if ":" in err else 60
            return {"error":True,"code":"RATE_LIMIT","message":f"Rate limit — wait {wait}s.",
                    "hint":f"⏳ Wait {wait} seconds then try again.","wait_seconds":wait}
        if "API_KEY_LEAKED" in err:
            return {"error":True,"code":"API_KEY_LEAKED",
                    "message":"Your Gemini API key was flagged as leaked.",
                    "hint":"Generate a new key at https://aistudio.google.com/apikey and update it in Render Environment."}
        return {"error":True,"code":"SQL_GEN_FAILED","message":f"Could not generate SQL: {err[:200]}",
                "hint":"Try rephrasing. Example: Show total revenue by product category"}

    # 4. Execute SQL
    try:
        result_df = run_sql(df_global, sql)
    except Exception as e:
        return {"error":True,"code":"SQL_EXEC_FAILED","message":f"Query failed: {str(e)[:200]}",
                "hint":"Try a simpler question.","sql":sql}

    if result_df is None or result_df.empty:
        return {"sql":sql,"rows":0,"columns":[],"data":[],"chart_b64":None,"chart_meta":None,"stats":None}

    # 5. Choose chart (uses Gemini — check rate limit again)
    chart_meta = choose_chart(sql, result_df.columns.tolist(), len(result_df))

    # 6. Render chart
    chart_b64 = None
    try:
        chart_b64 = render_chart_to_b64(result_df, chart_meta)
    except Exception as ce:
        print(f"Chart render error: {ce}")

    # 7. Stats
    y_col  = chart_meta.get("y_col", result_df.columns[-1])
    x_col  = chart_meta.get("x_col", result_df.columns[0])
    y_vals = pd.to_numeric(result_df[y_col], errors="coerce").dropna()
    stats  = None
    if len(y_vals) > 0:
        mi = y_vals.idxmax(); ni = y_vals.idxmin()
        stats = {"total":float(y_vals.sum()),"average":float(y_vals.mean()),
                 "max_val":float(y_vals.max()),"min_val":float(y_vals.min()),
                 "max_label":str(result_df.loc[mi,x_col]) if x_col in result_df.columns else "",
                 "min_label":str(result_df.loc[ni,x_col]) if x_col in result_df.columns else ""}

    # 8. Serialize
    data = []
    for row in result_df.head(200).to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            try:
                if pd.isna(v): clean[k]=None; continue
            except Exception: pass
            clean[k] = v.item() if hasattr(v,"item") else (str(v)[:10] if hasattr(v,"date") else v)
        data.append(clean)

    result = {"sql":sql,"rows":len(result_df),"columns":result_df.columns.tolist(),
              "data":data,"chart_b64":chart_b64,"chart_meta":chart_meta,"stats":stats}

    # 9. Cache result
    _cache_set(req.question, result)
    return result


@app.post("/chat")
def chat(req: ChatRequest):
    if not GEMINI_KEY:
        raise HTTPException(400, "GEMINI_API_KEY not set.")

    # Check rate limit
    can, reason, wait = rate_limiter.can_call()
    if not can:
        return {"reply":f"⏳ Rate limit reached. Please wait {wait} seconds and try again.",
                "rate_limited":True,"wait_seconds":wait}

    stats = compute_dataset_stats(df_global) if df_global is not None else {}
    conv  = [CHAT_SYSTEM, "\n\nCONVERSATION:"]
    for msg in req.history[-10:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        conv.append(f"\n{role}: {msg.get('content','')}")
    conv.append(f"\nUser: {req.message}\nAssistant:")

    try:
        reply = _call_gemini("\n".join(conv))
        return {"reply":reply,"status":"ok"}
    except Exception as e:
        err = str(e)
        if "RATE_LIMIT" in err:
            wait = int(err.split(":")[1]) if ":" in err else 60
            return {"reply":f"⏳ Rate limit reached. Please wait {wait} seconds.",
                    "rate_limited":True,"wait_seconds":wait}
        raise HTTPException(500, f"AI error: {err}")


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════
@app.on_event("startup")
def startup():
    global df_global, rag_chunks, filename

    # Init Supabase
    db_ok = init_db()

    # Load CSV into memory for fallback + RAG
    default_csv = DATA_DIR / "Amazon_Sales.csv"
    if default_csv.exists():
        try:
            df_global = pd.read_csv(str(default_csv))
            for col in df_global.columns:
                if "date" in col.lower():
                    df_global[col] = pd.to_datetime(df_global[col], errors="coerce")
            filename   = "Amazon_Sales.csv"
            rag_chunks = build_rag_chunks(df_global, filename)
            s = compute_dataset_stats(df_global)
            print(f"[startup] ✅ Loaded {len(df_global):,} rows from CSV")
            print(f"[startup]    Revenue: ${s.get('total_revenue',0):,.2f}")
            print(f"[startup]    Growth:  {s.get('growth_pct',0):+.2f}%")
            print(f"[startup]    DB:      {'Supabase ✅' if db_ok else 'pandas fallback'}")
            print(f"[startup]    Rate limit: {rate_limiter.MAX_PER_MIN}/min, {rate_limiter.MAX_PER_DAY}/day")
        except Exception as e:
            print(f"[startup] ❌ CSV load failed: {e}")
