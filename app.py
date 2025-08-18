import os
import io
import base64
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from

# ---------------------- Flask Setup ----------------------
app = Flask(__name__)
swagger = Swagger(app)

# ---------------------- Optional LLM ----------------------
USE_LLM = False
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

if USE_LLM and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
    except Exception:
        USE_LLM = False

# ---------------------- Helpers ----------------------
def read_csv_file(file_obj):
    try:
        return pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

def read_txt_file(file_obj):
    try:
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        return content.strip()
    except Exception as e:
        raise ValueError(f"Error reading TXT: {e}")

def fig_to_base64_png(fig, *, max_bytes=100_000, dpi=120):
    """Return RAW base64 (no data URI!) ensuring size < max_bytes by tuning dpi."""
    for attempt_dpi in [dpi, 110, 100, 90, 80]:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=attempt_dpi)
        plt.close(fig)
        raw = buf.getvalue()
        if len(raw) <= max_bytes:
            return base64.b64encode(raw).decode("utf-8")
    # fallback compress
    try:
        from PIL import Image
        buf_in = io.BytesIO(raw)
        img = Image.open(buf_in).convert("P", palette=Image.ADAPTIVE)
        buf_out = io.BytesIO()
        img.save(buf_out, format="PNG", optimize=True)
        return base64.b64encode(buf_out.getvalue()).decode("utf-8")
    except Exception:
        return base64.b64encode(raw).decode("utf-8")

# ---------------------- Analyzers ----------------------
def analyze_sample_sales(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    required = ["date", "region", "sales"]
    for k in required:
        if k not in cols:
            raise ValueError(f"CSV missing required column '{k}'")
    c_date, c_region, c_sales = cols["date"], cols["region"], cols["sales"]

    dfx = df.copy()
    dfx[c_date] = pd.to_datetime(dfx[c_date], errors="coerce")
    dfx[c_sales] = pd.to_numeric(dfx[c_sales], errors="coerce")
    dfx = dfx.dropna(subset=[c_date, c_region, c_sales])

    total_sales = float(dfx[c_sales].sum())
    by_region = dfx.groupby(c_region, dropna=False)[c_sales].sum().sort_values(ascending=False)
    top_region = str(by_region.index[0]) if not by_region.empty else ""

    day_of_month = dfx[c_date].dt.day
    if day_of_month.nunique() > 1 and dfx[c_sales].nunique() > 1:
        day_sales_correlation = float(day_of_month.corr(dfx[c_sales]))
    else:
        day_sales_correlation = float("nan")

    fig1, ax1 = plt.subplots(figsize=(5, 3))
    by_region.plot(kind="bar", ax=ax1, color="blue")
    ax1.set_title("Total Sales by Region")
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Total Sales")
    fig1.tight_layout()
    bar_chart_b64 = fig_to_base64_png(fig1)

    median_sales = float(dfx[c_sales].median()) if len(dfx) else 0.0
    total_sales_tax = float(total_sales * 0.10)

    by_date = dfx.groupby(c_date)[c_sales].sum().sort_index()
    cum = by_date.cumsum()
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.plot(cum.index, cum.values, color="red")
    ax2.set_title("Cumulative Sales Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Sales")
    fig2.autofmt_xdate()
    fig2.tight_layout()
    cumulative_sales_chart_b64 = fig_to_base64_png(fig2)

    return {
        "total_sales": total_sales,
        "top_region": top_region,
        "day_sales_correlation": day_sales_correlation,
        "bar_chart": bar_chart_b64,
        "median_sales": median_sales,
        "total_sales_tax": total_sales_tax,
        "cumulative_sales_chart": cumulative_sales_chart_b64,
    }

def analyze_sample_weather(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    c_temp = next((cols[c] for c in cols if "temp" in c), None)
    c_precip = next((cols[c] for c in cols if "precip" in c), None)
    c_date = next((cols[c] for c in cols if "date" in c or "day" in c), None)

    if not c_temp or not c_precip:
        raise ValueError("CSV missing required columns for weather (need temp & precip)")

    dfx = df.copy()
    dfx[c_temp] = pd.to_numeric(dfx[c_temp], errors="coerce")
    dfx[c_precip] = pd.to_numeric(dfx[c_precip], errors="coerce")

    avg_temp = float(dfx[c_temp].mean())
    min_temp = float(dfx[c_temp].min())
    avg_precip = float(dfx[c_precip].mean())
    corr = float(dfx[c_temp].corr(dfx[c_precip]))
    max_precip_date = str(dfx.loc[dfx[c_precip].idxmax(), c_date]) if c_date else None

    return {
        "average_temp_c": avg_temp,
        "min_temp_c": min_temp,
        "average_precip_mm": avg_precip,
        "temp_precip_correlation": corr,
        "max_precip_date": max_precip_date,
    }

def analyze_graph(df: pd.DataFrame):
    u, v = df.columns[:2]
    edges = df[[u, v]].dropna().astype(str).values.tolist()

    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    edge_count = len(edges)
    degrees = {node: len(neigh) for node, neigh in adj.items()}
    highest_degree_node = max(degrees, key=degrees.get)
    avg_degree = sum(degrees.values()) / len(degrees)
    n = len(adj)
    density = 2 * edge_count / (n * (n - 1)) if n > 1 else 0

    def bfs(start, goal):
        from collections import deque
        q = deque([(start, 0)])
        seen = {start}
        while q:
            node, dist = q.popleft()
            if node == goal:
                return dist
            for neigh in adj.get(node, []):
                if neigh not in seen:
                    seen.add(neigh)
                    q.append((neigh, dist + 1))
        return None

    return {
        "edge_count": edge_count,
        "highest_degree_node": highest_degree_node,
        "average_degree": avg_degree,
        "density": density,
        "shortest_path_alice_bob": bfs("alice", "bob"),
    }

# ---------------------- Task Detection ----------------------
def detect_task(questions_text: str) -> str:
    qt = questions_text.lower()
    if "sample-sales" in qt:
        return "sample_sales"
    if "sample-weather" in qt:
        return "sample_weather"
    if "edges.csv" in qt or "graph" in qt or "network" in qt:
        return "graph"
    return "generic"

# ---------------------- Routes ----------------------
@app.route("/", methods=["GET"])
def index():
    return "Welcome! Use POST /api for file upload & querying. Swagger docs at /apidocs"

@app.route("/api", methods=["POST"])
@swag_from({
    "tags": ["Data Analyst Agent API"],
    "consumes": ["multipart/form-data"],
    "parameters": [
        {"name": "questions.txt", "in": "formData", "type": "file", "required": False},
        {"name": "questions_file", "in": "formData", "type": "file", "required": False},
        {"name": "data.csv", "in": "formData", "type": "file", "required": False},
        {"name": "csv_file", "in": "formData", "type": "file", "required": False},
        {"name": "query", "in": "formData", "type": "string", "required": False}
    ],
    "responses": {200: {"description": "JSON object"}}
})
def unified_api():
    questions_file = request.files.get("questions.txt") or request.files.get("questions_file")
    csv_file = request.files.get("data.csv") or request.files.get("csv_file")
    query_text = request.form.get("query")

    if not questions_file and not query_text:
        return jsonify({"error": "questions.txt (or 'query') is required"}), 400

    questions_text = query_text or read_txt_file(questions_file)
    task = detect_task(questions_text)

    if not csv_file:
        return jsonify({"error": f"Task '{task}' requires a CSV file"}), 400

    try:
        df = read_csv_file(csv_file)
        if task == "sample_sales":
            return jsonify(analyze_sample_sales(df)), 200
        elif task == "sample_weather":
            return jsonify(analyze_sample_weather(df)), 200
        elif task == "graph":
            return jsonify(analyze_graph(df)), 200
        else:
            return jsonify({"error": "Unrecognized task. Please clarify in questions.txt"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to analyze: {e}"}), 500

# ---------------------- Run ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
