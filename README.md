# AI Data Cosmos Explorer — VS Code Setup

Full-stack AI Sales Analytics app.
Frontend: Original AI Data Cosmos HTML (unchanged)
Backend:  FastAPI + Gemini AI + Pandas/DuckDB

## Project Structure

```
ai_sales_vscode/
├── backend/
│   ├── __init__.py
│   └── main.py          ← FastAPI server (all logic from Colab notebook)
├── frontend/
│   └── index.html       ← Original AI Data Cosmos frontend (unchanged)
├── data/
│   └── Amazon_Sales.csv ← Auto-loaded on startup
├── .env                 ← Your Gemini API key goes here
├── requirements.txt
└── README.md
```

---

## Setup in VS Code

### Step 1 — Open the folder
File → Open Folder → select `ai_sales_vscode`

### Step 2 — Create a Python virtual environment
Open the VS Code terminal (Ctrl+`)

```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Add your Gemini API key
Open `.env` and replace:
```
GEMINI_API_KEY=paste_your_gemini_key_here
```
With your real key from https://aistudio.google.com/

### Step 5 — Start the backend server
```bash
uvicorn backend.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
[startup] ✅ Loaded 50,000 rows from ...Amazon_Sales.csv
```

### Step 6 — Open the frontend
Open `frontend/index.html` in your browser.

Option A — VS Code Live Server extension:
- Install "Live Server" extension in VS Code
- Right-click `index.html` → "Open with Live Server"

Option B — Direct file:
- Double-click `frontend/index.html`

---

## How to Use

1. The page loads with the cosmos galaxy background
2. Click **LOAD: SALES DATA** 
3. Click **INITIATE DEEP SCAN**
4. The Analytics HUD opens — use the **LIVE QUERY TERMINAL**
5. Click any quick chip or type a question → press **▶ EXECUTE**
6. See: Generated SQL → Stat cards → Interactive chart → Data table

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Serves frontend |
| GET | `/health` | Backend + dataset status |
| GET | `/schema` | Dataset schema |
| POST | `/upload` | Upload a new CSV |
| POST | `/query` | NL → SQL → Chart |

---

## Sample Questions

- "Show monthly total revenue for 2023"
- "Top 5 product categories by revenue"
- "Sales by customer region"
- "Average rating by product category"
- "Revenue by payment method"
- "Top 10 highest revenue months"
- "Monthly quantity sold trend"
- "Compare discount percent by region"

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `GEMINI_API_KEY not set` | Edit `.env` file |
| Chart not showing | Check browser console for errors |
| `Connection refused` | Run `uvicorn backend.main:app --reload` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `429 rate limit` | Wait 60s, Gemini retries automatically |
