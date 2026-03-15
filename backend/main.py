# ═══════════════════════════════════════════════════════════════════════════
# AI Sales Data Analyst — FastAPI Backend v2.0
# Run: uvicorn backend.main:app --reload --port 8000
# ═══════════════════════════════════════════════════════════════════════════

import re, json, io, base64, warnings, time, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path
from dotenv import load_dotenv

import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

warnings.filterwarnings("ignore")
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.5-flash"
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
PORT       = int(os.getenv("PORT", "8080"))
# Works both locally and in Cloud Run (/app/)
_BASE    = Path(__file__).parent.parent
DATA_DIR = _BASE / "data"
FRONT_DIR= _BASE / "frontend"

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
# DATASET STATS  — reads full CSV and computes everything
# ═══════════════════════════════════════════════════════════════════════════
def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute complete dataset statistics including growth factor."""
    stats = {}

    # Basic counts
    stats["total_rows"]    = len(df)
    stats["total_columns"] = len(df.columns)
    stats["columns"]       = df.columns.tolist()

    # Date range
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols:
        dc = date_cols[0]
        try:
            dates = pd.to_datetime(df[dc], errors="coerce").dropna()
            stats["date_min"]   = str(dates.min())[:10]
            stats["date_max"]   = str(dates.max())[:10]
            stats["date_min_m"] = str(dates.min())[:7]
            stats["date_max_m"] = str(dates.max())[:7]
            stats["total_months"] = len(dates.dt.to_period("M").unique())
            stats["total_years"]  = sorted(dates.dt.year.unique().tolist())
        except Exception:
            pass

    # Revenue stats
    if "total_revenue" in df.columns:
        rev = pd.to_numeric(df["total_revenue"], errors="coerce").dropna()
        stats["total_revenue"] = float(rev.sum())
        stats["avg_revenue"]   = float(rev.mean())
        stats["max_revenue"]   = float(rev.max())
        stats["min_revenue"]   = float(rev.min())

    # YoY Growth factor
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
                stats["growth_pct"]  = round(((y2 - y1) / y1) * 100, 2) if y1 else 0
                stats["growth_y1"]   = years[-2]
                stats["growth_y2"]   = years[-1]
                stats["revenue_y1"]  = round(y1, 2)
                stats["revenue_y2"]  = round(y2, 2)

            # Monthly growth (first 3 vs last 3 months avg)
            monthly = df2.groupby(df2[dc].dt.to_period("M"))["total_revenue"].sum()
            if len(monthly) >= 6:
                first3 = float(monthly.head(3).mean())
                last3  = float(monthly.tail(3).mean())
                stats["monthly_growth_pct"] = round(((last3 - first3) / first3) * 100, 2) if first3 else 0
                stats["best_month"]  = str(monthly.idxmax())
                stats["best_month_rev"] = round(float(monthly.max()), 2)
                stats["worst_month"] = str(monthly.idxmin())
        except Exception as e:
            print(f"Growth calc error: {e}")

    # Category breakdown
    if "product_category" in df.columns and "total_revenue" in df.columns:
        try:
            cats = df.groupby("product_category")["total_revenue"].sum().sort_values(ascending=False)
            stats["categories"] = {k: round(float(v), 2) for k, v in cats.items()}
        except Exception:
            pass

    # Region breakdown
    if "customer_region" in df.columns and "total_revenue" in df.columns:
        try:
            regs = df.groupby("customer_region")["total_revenue"].sum().sort_values(ascending=False)
            stats["regions"] = {k: round(float(v), 2) for k, v in regs.items()}
        except Exception:
            pass

    # Quantity sold
    if "quantity_sold" in df.columns:
        try:
            stats["total_quantity"] = int(df["quantity_sold"].sum())
        except Exception:
            pass

    # Avg rating
    if "rating" in df.columns:
        try:
            stats["avg_rating"] = round(float(df["rating"].mean()), 2)
        except Exception:
            pass

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# SQL EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════
def run_sql(df_src, sql):
    try:
        import duckdb
        df = df_src.copy()
        return duckdb.query(sql.replace("sales_data", "df")).df()
    except Exception:
        pass
    return _pandas_fallback(df_src, sql)


def _pandas_fallback(df_src, sql):
    df = df_src.copy()
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

    AGG = {"SUM": "sum", "COUNT": "count", "AVG": "mean", "MIN": "min", "MAX": "max"}
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
                found = next((alias for alias, _, _, _ in select_items if alias.lower() == gc.lower()), None)
                if found and found in df.columns:
                    group_cols.append(found)
        agg_specs = {}
        for alias, src, func, _ in select_items:
            if func and src in df.columns:
                agg_specs[src] = (alias, func)
        if group_cols and agg_specs:
            agg_dict = {src: func for src, (_, func) in agg_specs.items()}
            try:
                grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
                rename  = {src: alias for src, (alias, _) in agg_specs.items()}
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
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
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
# GEMINI ENGINE
# ═══════════════════════════════════════════════════════════════════════════
SQL_PROMPT = """You are an expert DuckDB SQL analyst.

DATASET CONTEXT:
{context}

FULL SCHEMA:
{schema}

USER QUESTION:
{question}

RULES:
1. Generate ONLY a single SELECT query.
2. Table name is ALWAYS: sales_data
3. For date operations: strftime(order_date, '%Y-%m') for month, strftime(order_date, '%Y') for year.
4. For top N: use ORDER BY ... DESC LIMIT N.
5. Exact column names: order_date, product_id, product_category, price, discount_percent,
   quantity_sold, customer_region, payment_method, rating, review_count, discounted_price, total_revenue
6. Return ONLY the SQL query — no explanation, no markdown fences.

SQL QUERY:"""

CHART_PROMPT = """Given this SQL and result columns, pick the best chart type.
SQL: {sql}
Result columns: {columns}
Row count: {row_count}
Return ONLY a JSON object with these exact keys:
- chart_type: one of line, bar, horizontal_bar, pie, donut, area, scatter
- x_col: the column name to use for x-axis labels
- y_col: the column name to use for y-axis values
- title: short descriptive title
- reason: one sentence why this chart type
No markdown, no extra text.
JSON:"""

_BLOCKED = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|EXEC)\b",
    re.IGNORECASE
)


def _call_gemini(prompt, retries=3):
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(retries):
        try:
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"Rate limit — waiting 60s...")
                time.sleep(60)
            else:
                raise e


def generate_sql(question, df, chunks):
    context = retrieve_context(chunks, question)
    schema  = get_schema_text(df)
    prompt  = SQL_PROMPT.format(context=context, schema=schema, question=question)
    raw     = _call_gemini(prompt)
    raw     = re.sub(r"```sql|```", "", raw, flags=re.IGNORECASE)
    m       = re.search(r"(SELECT\b.*)", raw, re.IGNORECASE | re.DOTALL)
    sql     = m.group(1).strip() if m else raw.strip()
    if _BLOCKED.search(sql) or not sql.upper().startswith("SELECT"):
        raise ValueError(f"Unsafe SQL blocked: {sql[:80]}")
    return sql


def choose_chart(sql, columns, row_count):
    try:
        prompt = CHART_PROMPT.format(sql=sql, columns=columns, row_count=row_count)
        raw    = re.sub(r"```json|```", "", _call_gemini(prompt)).strip()
        # Handle cases where Gemini wraps in extra text
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        meta = json.loads(raw)
        if meta.get("x_col") not in columns:
            meta["x_col"] = columns[0]
        if meta.get("y_col") not in columns:
            num_cols      = [c for c in columns if c != meta.get("x_col")]
            meta["y_col"] = num_cols[-1] if num_cols else columns[-1]
        return meta
    except Exception:
        col_lower = " ".join(columns).lower()
        if any(k in col_lower for k in ["month", "year", "date", "time"]):
            return {"chart_type": "area", "x_col": columns[0], "y_col": columns[-1],
                    "title": "Trend Over Time", "reason": "Time series data."}
        if row_count <= 6:
            return {"chart_type": "donut", "x_col": columns[0], "y_col": columns[-1],
                    "title": "Distribution", "reason": "Few categories."}
        if row_count <= 10:
            return {"chart_type": "horizontal_bar", "x_col": columns[0], "y_col": columns[-1],
                    "title": "Comparison", "reason": "Category comparison."}
        return {"chart_type": "bar", "x_col": columns[0], "y_col": columns[-1],
                "title": "Comparison", "reason": "Bar chart for comparisons."}


# ═══════════════════════════════════════════════════════════════════════════
# CHART ENGINE — Fixed, clean, readable charts
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v):
    """Format number to short readable string."""
    try:
        v = float(v)
        if abs(v) >= 1_000_000: return f"{v/1e6:.1f}M"
        if abs(v) >= 1_000:     return f"{v/1e3:.0f}K"
        return f"{v:,.1f}" if v != int(v) else f"{v:,.0f}"
    except Exception:
        return str(v)


def _base_fig(w=12, h=5):
    """Create a styled figure."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    return fig, ax


def _style_ax(ax, title, xlabel="", ylabel=""):
    """Apply consistent dark styling to axes."""
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold",
                 pad=14, loc="left", fontfamily="monospace")
    ax.tick_params(colors=MUTED, labelsize=9, length=3)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--", alpha=0.8)
    ax.xaxis.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9, labelpad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt(x)))


def _to_b64(fig):
    """Save figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _resolve_cols(result_df, chart_meta):
    """Safely resolve x_col and y_col, auto-detect numeric y."""
    cols   = result_df.columns.tolist()
    x_col  = chart_meta.get("x_col")
    y_col  = chart_meta.get("y_col")

    if not x_col or x_col not in cols:
        x_col = cols[0]
    if not y_col or y_col not in cols:
        num_cols = result_df.select_dtypes(include="number").columns.tolist()
        y_col    = next((c for c in num_cols if c != x_col), cols[-1])

    return x_col, y_col


def render_bar(result_df, chart_meta):
    x_col, y_col = _resolve_cols(result_df, chart_meta)
    title = chart_meta.get("title", "Result")

    # Limit to top 20
    if len(result_df) > 20:
        result_df = result_df.head(20)
        title += " (Top 20)"

    x_vals = result_df[x_col].astype(str).tolist()
    y_vals = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0).tolist()

    fig, ax = _base_fig(max(10, len(x_vals) * 0.8), 5)
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(x_vals))]

    bars = ax.bar(range(len(x_vals)), y_vals, color=colors,
                  edgecolor=BG, linewidth=0.5, width=0.65, zorder=2)

    # Value labels on top of bars
    max_y = max(y_vals) if y_vals else 1
    for bar, val in zip(bars, y_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_y * 0.015,
                    _fmt(val), ha="center", va="bottom",
                    fontsize=8, color=TEXT, fontweight="600")

    # X tick labels — category names
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals, rotation=30, ha="right", fontsize=9, color=TEXT)
    ax.set_xlim(-0.6, len(x_vals) - 0.4)
    _style_ax(ax, title, x_col.replace("_", " ").title(),
              y_col.replace("_", " ").title())
    fig.tight_layout(pad=1.6)
    return _to_b64(fig)


def render_horizontal_bar(result_df, chart_meta):
    x_col, y_col = _resolve_cols(result_df, chart_meta)
    title = chart_meta.get("title", "Result")

    if len(result_df) > 20:
        result_df = result_df.head(20)

    x_vals = result_df[x_col].astype(str).tolist()
    y_vals = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0).tolist()

    # Reverse so largest is on top
    x_rev = list(reversed(x_vals))
    y_rev = list(reversed(y_vals))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(x_rev))]

    fig, ax = _base_fig(11, max(4, len(x_rev) * 0.5))
    bars = ax.barh(range(len(x_rev)), y_rev,
                   color=list(reversed(colors)),
                   edgecolor=BG, linewidth=0.5, height=0.6, zorder=2)

    max_y = max(y_rev) if y_rev else 1
    for bar, val in zip(bars, y_rev):
        ax.text(bar.get_width() + max_y * 0.01,
                bar.get_y() + bar.get_height() / 2,
                _fmt(val), va="center", ha="left", fontsize=8.5, color=TEXT)

    ax.set_yticks(range(len(x_rev)))
    ax.set_yticklabels(x_rev, fontsize=9, color=TEXT)
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold",
                 pad=14, loc="left", fontfamily="monospace")
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: _fmt(x)))
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, linestyle="--")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.6)
    return _to_b64(fig)


def render_line(result_df, chart_meta):
    x_col, y_col = _resolve_cols(result_df, chart_meta)
    title = chart_meta.get("title", "Trend")

    x_vals = result_df[x_col].astype(str).tolist()
    y_vals = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0)

    fig, ax = _base_fig(13, 5)
    x_idx   = range(len(x_vals))

    ax.plot(x_idx, y_vals, color=PALETTE[0], linewidth=2.5,
            marker="o", markersize=5,
            markerfacecolor=BG, markeredgecolor=PALETTE[0],
            markeredgewidth=2, zorder=3, label=y_col.replace("_", " ").title())

    ax.fill_between(x_idx, y_vals, alpha=0.10, color=PALETTE[0])

    # Annotate peak
    if len(y_vals) > 0:
        peak_i = int(np.argmax(y_vals))
        ax.annotate(
            f"Peak\n{_fmt(y_vals.iloc[peak_i])}",
            xy=(peak_i, y_vals.iloc[peak_i]),
            xytext=(0, 18), textcoords="offset points",
            ha="center", fontsize=8, color=PALETTE[0], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=1.2)
        )

    # Rolling average
    if len(y_vals) >= 4:
        roll = pd.Series(y_vals).rolling(3, center=True).mean()
        ax.plot(x_idx, roll, color=PALETTE[1], linewidth=1.5,
                linestyle="--", alpha=0.7, label="3-pt avg")
        ax.legend(facecolor=CARD2, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=8, loc="upper left")

    # X labels — show every 2nd if too many
    step = max(1, len(x_vals) // 12)
    ax.set_xticks(range(0, len(x_vals), step))
    ax.set_xticklabels(x_vals[::step], rotation=40, ha="right", fontsize=8.5, color=TEXT)
    _style_ax(ax, title, x_col.replace("_", " ").title(),
              y_col.replace("_", " ").title())
    fig.tight_layout(pad=1.6)
    return _to_b64(fig)


def render_area(result_df, chart_meta):
    x_col, y_col = _resolve_cols(result_df, chart_meta)
    title = chart_meta.get("title", "Trend")

    x_vals = result_df[x_col].astype(str).tolist()
    y_vals = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0)

    fig, ax = _base_fig(13, 5)
    x_idx   = range(len(x_vals))

    ax.fill_between(x_idx, y_vals, alpha=0.22, color=PALETTE[0])
    ax.plot(x_idx, y_vals, color=PALETTE[0], linewidth=2.5, zorder=3)

    if len(y_vals) >= 4:
        roll = pd.Series(y_vals).rolling(3, center=True).mean()
        ax.plot(x_idx, roll, color=PALETTE[1], linewidth=1.8,
                linestyle="--", alpha=0.8, label="3-pt moving avg")
        ax.legend(facecolor=CARD2, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=8, loc="upper left")

    step = max(1, len(x_vals) // 12)
    ax.set_xticks(range(0, len(x_vals), step))
    ax.set_xticklabels(x_vals[::step], rotation=40, ha="right", fontsize=8.5, color=TEXT)
    _style_ax(ax, title, x_col.replace("_", " ").title(),
              y_col.replace("_", " ").title())
    fig.tight_layout(pad=1.6)
    return _to_b64(fig)


def render_pie(result_df, chart_meta):
    x_col, y_col = _resolve_cols(result_df, chart_meta)
    title = chart_meta.get("title", "Distribution")

    # Limit to top 8 slices
    if len(result_df) > 8:
        result_df = result_df.head(8)

    labels = result_df[x_col].astype(str).tolist()
    values = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0).tolist()
    colors = PALETTE[:len(labels)]

    fig, ax = _base_fig(9, 6)

    wedges, texts, autotexts = ax.pie(
        values, labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90,
        wedgeprops={"edgecolor": BG, "linewidth": 2},
        textprops={"color": TEXT, "fontsize": 9}
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
        at.set_color(TEXT)

    # Legend with names and values
    legend_labels = [f"{lbl}  ({_fmt(val)})" for lbl, val in zip(labels, values)]
    ax.legend(wedges, legend_labels,
              loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(3, len(labels)),
              facecolor=CARD2, edgecolor=BORDER,
              labelcolor=TEXT, fontsize=8.5, framealpha=0.9)

    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold",
                 pad=14, fontfamily="monospace")
    fig.tight_layout(pad=1.6)
    return _to_b64(fig)


def render_donut(result_df, chart_meta):
    x_col, y_col = _resolve_cols(result_df, chart_meta)
    title = chart_meta.get("title", "Distribution")

    if len(result_df) > 8:
        result_df = result_df.head(8)

    labels = result_df[x_col].astype(str).tolist()
    values = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0).tolist()
    colors = PALETTE[:len(labels)]
    total  = sum(values)

    fig, ax = _base_fig(9, 6)

    wedges, _, autotexts = ax.pie(
        values, labels=None,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90,
        wedgeprops={"edgecolor": BG, "linewidth": 2, "width": 0.52},
        textprops={"color": TEXT, "fontsize": 9}
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_fontweight("bold")
        at.set_color(TEXT)

    # Centre total
    ax.text(0, 0, f"{_fmt(total)}\nTotal",
            ha="center", va="center",
            fontsize=12, color=TEXT, fontweight="bold", linespacing=1.6)

    legend_labels = [f"{lbl}  ({_fmt(val)})" for lbl, val in zip(labels, values)]
    ax.legend(wedges, legend_labels,
              loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(3, len(labels)),
              facecolor=CARD2, edgecolor=BORDER,
              labelcolor=TEXT, fontsize=8.5, framealpha=0.9)

    ax.set_title(title, color=TEXT, fontsize=13, fontweight="bold",
                 pad=14, fontfamily="monospace")
    fig.tight_layout(pad=1.6)
    return _to_b64(fig)


def render_scatter(result_df, chart_meta):
    x_col, y_col = _resolve_cols(result_df, chart_meta)
    title = chart_meta.get("title", "Scatter")

    x_vals = result_df[x_col].astype(str).tolist()
    y_vals = pd.to_numeric(result_df[y_col], errors="coerce").fillna(0).tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(x_vals))]

    fig, ax = _base_fig(12, 5)
    ax.scatter(range(len(x_vals)), y_vals, color=colors,
               s=90, alpha=0.85, edgecolors=BG, linewidth=0.8, zorder=3)

    step = max(1, len(x_vals) // 12)
    ax.set_xticks(range(0, len(x_vals), step))
    ax.set_xticklabels(x_vals[::step], rotation=35, ha="right", fontsize=9, color=TEXT)
    _style_ax(ax, title, x_col.replace("_", " ").title(),
              y_col.replace("_", " ").title())
    fig.tight_layout(pad=1.6)
    return _to_b64(fig)


def render_chart_to_b64(result_df, chart_meta):
    """Dispatch to the correct chart renderer."""
    ctype = chart_meta.get("chart_type", "bar").lower().strip()

    dispatch = {
        "bar":            render_bar,
        "horizontal_bar": render_horizontal_bar,
        "line":           render_line,
        "area":           render_area,
        "pie":            render_pie,
        "donut":          render_donut,
        "scatter":        render_scatter,
    }

    renderer = dispatch.get(ctype, render_bar)
    return renderer(result_df.copy(), chart_meta)


# ═══════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="AI Sales Data Analyst", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONT_DIR / "index.html"))


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if df_global is None:
        return {"status": "ok", "dataset_loaded": False, "rows": 0,
                "revenue": 0, "model": MODEL_NAME}
    try:
        rev = float(df_global["total_revenue"].sum()) if "total_revenue" in df_global.columns else 0
    except Exception:
        rev = 0
    return {
        "status":         "ok",
        "dataset_loaded": True,
        "filename":       filename,
        "rows":           len(df_global),
        "revenue":        rev,
        "model":          MODEL_NAME,
    }


# ── Dataset stats (full computed stats) ──────────────────────────────────────
@app.get("/stats")
def get_stats():
    if df_global is None:
        raise HTTPException(400, "No dataset loaded.")
    return compute_dataset_stats(df_global)


# ── Schema ────────────────────────────────────────────────────────────────────
@app.get("/schema")
def schema():
    if df_global is None:
        raise HTTPException(400, "No dataset loaded.")
    stats = compute_dataset_stats(df_global)
    return {
        "filename": filename,
        "rows":     len(df_global),
        "columns":  df_global.columns.tolist(),
        "schema":   get_schema_text(df_global),
        "stats":    stats,
    }


# ── Upload CSV ────────────────────────────────────────────────────────────────
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
    stats      = compute_dataset_stats(df_global)
    return {
        "success":  True,
        "filename": filename,
        "rows":     len(df_global),
        "columns":  df_global.columns.tolist(),
        "stats":    stats,
        "preview":  df_global.head(5).to_dict(orient="records"),
    }


# ── Query ─────────────────────────────────────────────────────────────────────
@app.post("/query")
def query(req: QueryRequest):
    global df_global, rag_chunks
    if df_global is None:
        return {"error":True,"code":"NO_DATA","message":"No dataset loaded.","hint":"Check terminal — server auto-loads Amazon_Sales.csv on startup."}
    if not GEMINI_KEY:
        return {"error":True,"code":"NO_KEY","message":"Gemini API key not set.","hint":"Open .env and set GEMINI_API_KEY=your_key_here"}

    # 1. Generate SQL
    sql = None
    try:
        sql = generate_sql(req.question, df_global, rag_chunks)
    except Exception as e:
        err = str(e)
        if "429" in err:
            return {"error":True,"code":"RATE_LIMIT","message":"Gemini rate limit reached.","hint":"Wait 60 seconds then try again."}
        return {"error":True,"code":"SQL_GEN_FAILED","message":f"Could not generate SQL: {err[:200]}","hint":"Try rephrasing. E.g. Show total revenue by product category"}

    # 2. Execute
    try:
        result_df = run_sql(df_global, sql)
    except Exception as e:
        return {"error":True,"code":"SQL_EXEC_FAILED","message":f"Query failed: {str(e)[:200]}","hint":"Try a simpler question. E.g. Top 5 categories by revenue","sql":sql}

    if result_df is None or result_df.empty:
        return {"sql": sql, "rows": 0, "columns": [], "data": [],
                "chart_b64": None, "chart_meta": None, "stats": None}

    # 3. Choose chart
    chart_meta = choose_chart(sql, result_df.columns.tolist(), len(result_df))

    # 4. Render chart
    chart_b64 = None
    try:
        chart_b64 = render_chart_to_b64(result_df, chart_meta)
    except Exception as ce:
        print(f"Chart render error: {ce}")

    # 5. Compute result stats
    y_col  = chart_meta.get("y_col", result_df.columns[-1])
    x_col  = chart_meta.get("x_col", result_df.columns[0])
    y_vals = pd.to_numeric(result_df[y_col], errors="coerce").dropna()
    stats  = None
    if len(y_vals) > 0:
        mi    = y_vals.idxmax()
        ni    = y_vals.idxmin()
        stats = {
            "total":     float(y_vals.sum()),
            "average":   float(y_vals.mean()),
            "max_val":   float(y_vals.max()),
            "min_val":   float(y_vals.min()),
            "max_label": str(result_df.loc[mi, x_col]) if x_col in result_df.columns else "",
            "min_label": str(result_df.loc[ni, x_col]) if x_col in result_df.columns else "",
        }

    # 6. Serialize rows safely
    data = []
    for row in result_df.head(200).to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            try:
                if pd.isna(v):
                    clean[k] = None; continue
            except Exception:
                pass
            clean[k] = v.item() if hasattr(v, "item") else (
                str(v)[:10] if hasattr(v, "date") else v)
        data.append(clean)

    return {
        "sql":        sql,
        "rows":       len(result_df),
        "columns":    result_df.columns.tolist(),
        "data":       data,
        "chart_b64":  chart_b64,
        "chart_meta": chart_meta,
        "stats":      stats,
    }


# ── Startup: auto-load default CSV ────────────────────────────────────────────
@app.on_event("startup")
def startup():
    global df_global, rag_chunks, filename
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
            print(f"[startup] ✅ Loaded {len(df_global):,} rows")
            print(f"[startup]    Revenue: ${s.get('total_revenue',0):,.2f}")
            print(f"[startup]    Date range: {s.get('date_min','')} → {s.get('date_max','')}")
            print(f"[startup]    YoY Growth: {s.get('growth_pct',0):+.2f}%")
        except Exception as e:
            print(f"[startup] ❌ Could not load default CSV: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CHAT ENDPOINT  — normal LLM conversation about the data
# ═══════════════════════════════════════════════════════════════════════════
class ChatRequest(BaseModel):
    message: str
    history: list = []   # list of {role: "user"|"assistant", content: str}


CHAT_SYSTEM = """You are a helpful data analyst assistant for an Amazon Sales dataset.

DATASET FACTS (always use these when answering):
- File: Amazon_Sales.csv
- Total rows: {rows:,}
- Columns: order_date, product_id, product_category, price, discount_percent,
  quantity_sold, customer_region, payment_method, rating, review_count,
  discounted_price, total_revenue
- Date range: {date_min} to {date_max}
- Total revenue: ${total_revenue:,.2f}
- YoY growth: {growth_pct:+.2f}% ({growth_y1} vs {growth_y2})
- Product categories: Books, Fashion, Sports, Beauty, Electronics, Home & Kitchen
- Customer regions: North America, Asia, Europe, Middle East
- Payment methods: UPI, Credit Card, Wallet, Cash on Delivery, Debit Card
- Average rating: ~3.0 out of 5
- Top revenue region: Middle East ($8.3M)
- Top revenue category: Beauty ($5.55M)

You can:
1. Answer questions about the data in plain English
2. Explain trends, patterns and insights
3. Suggest what queries to run
4. Help interpret chart results
5. Give business recommendations based on the data

Be concise, friendly and data-driven. Use actual numbers from the dataset when possible.
Format responses with clear paragraphs. Use bullet points for lists.
"""


@app.post("/chat")
def chat(req: ChatRequest):
    """Natural language chat about the dataset — no SQL generation."""
    if not GEMINI_KEY:
        raise HTTPException(400, "GEMINI_API_KEY not set in .env file.")

    # Build system context with real dataset stats
    stats = compute_dataset_stats(df_global) if df_global is not None else {}
    system_ctx = CHAT_SYSTEM.format(
        rows         = stats.get("total_rows", 50000),
        date_min     = stats.get("date_min", "2022-01-01"),
        date_max     = stats.get("date_max", "2023-12-31"),
        total_revenue= stats.get("total_revenue", 32866573.74),
        growth_pct   = stats.get("growth_pct", 0.54),
        growth_y1    = stats.get("growth_y1", 2022),
        growth_y2    = stats.get("growth_y2", 2023),
    )

    # Build conversation prompt
    conv_parts = [system_ctx, "\n\nCONVERSATION HISTORY:"]
    for msg in req.history[-10:]:   # keep last 10 turns for context
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        prefix  = "User" if role == "user" else "Assistant"
        conv_parts.append(f"\n{prefix}: {content}")

    conv_parts.append(f"\nUser: {req.message}")
    conv_parts.append("\nAssistant:")

    full_prompt = "\n".join(conv_parts)

    try:
        genai.configure(api_key=GEMINI_KEY)
        model    = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(full_prompt)
        reply    = response.text.strip()
        return {"reply": reply, "status": "ok"}
    except Exception as e:
        err = str(e)
        if "429" in err:
            raise HTTPException(429, "Rate limit hit. Please wait 60 seconds and try again.")
        if "API_KEY" in err.upper() or "api key" in err.lower():
            raise HTTPException(401, "Invalid Gemini API key. Check your .env file.")
        raise HTTPException(500, f"AI error: {err}")
