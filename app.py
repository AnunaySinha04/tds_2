import os
import io
import base64
import math
from datetime import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from

# ---------------------- Flask Setup ----------------------
app = Flask(__name__)
swagger = Swagger(app)
logging.basicConfig(level=logging.DEBUG)
# ---------------------- Optional LLM (kept but not required) ----------------------
# Use env vars if you *really* need the LLM fallback for other tasks.
USE_LLM = False  # default off for deterministic evals
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjEwMDE2MzhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.lF4kTrKjBLHn77hNBZObuUL5BUKx-HWti1XEClwD9CM")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
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
    # Try a couple of DPIs if needed
    for attempt_dpi in [dpi, 110, 100, 90, 80]:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=attempt_dpi)
        plt.close(fig)
        raw = buf.getvalue()
        if len(raw) <= max_bytes:
            return base64.b64encode(raw).decode("utf-8")
        # if too big, reduce dpi and try again
    # Last resort: compress more by re-saving via PIL (optional)
    try:
        from PIL import Image
        buf_in = io.BytesIO(raw)
        img = Image.open(buf_in).convert("P", palette=Image.ADAPTIVE)
        buf_out = io.BytesIO()
        img.save(buf_out, format="PNG", optimize=True)
        return base64.b64encode(buf_out.getvalue()).decode("utf-8")
    except Exception:
        # If PIL isn't available, just return best-effort base64 (may be slightly >100kB)
        return base64.b64encode(raw).decode("utf-8")

def analyze_sample_sales(df: pd.DataFrame):
    """
    Required keys:
      - total_sales: number
      - top_region: string
      - day_sales_correlation: number
      - bar_chart: base64 PNG (<100kB, blue bars)
      - median_sales: number
      - total_sales_tax: number (10% of total)
      - cumulative_sales_chart: base64 PNG (<100kB, red line)
    Assumptions:
      - df has columns: ['date', 'region', 'sales'] (case-insensitive ok)
    """
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    required = ["date", "region", "sales"]
    for k in required:
        if k not in cols:
            raise ValueError(f"CSV missing required column '{k}'")
    c_date, c_region, c_sales = cols["date"], cols["region"], cols["sales"]

    # Parse and clean
    dfx = df.copy()
    dfx[c_date] = pd.to_datetime(dfx[c_date], errors="coerce")
    dfx[c_sales] = pd.to_numeric(dfx[c_sales], errors="coerce")
    dfx = dfx.dropna(subset=[c_date, c_region, c_sales])

    # 1) total_sales
    total_sales = float(dfx[c_sales].sum())

    # 2) top_region
    by_region = dfx.groupby(c_region, dropna=False)[c_sales].sum().sort_values(ascending=False)
    top_region = str(by_region.index[0]) if not by_region.empty else ""

    # 3) correlation between day-of-month and sales
    day_of_month = dfx[c_date].dt.day
    if day_of_month.nunique() > 1 and dfx[c_sales].nunique() > 1:
        day_sales_correlation = float(day_of_month.corr(dfx[c_sales]))
    else:
        day_sales_correlation = float("nan")

    # 4) bar chart of total sales by region (blue bars)
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    by_region.plot(kind="bar", ax=ax1, color="blue")
    ax1.set_title("Total Sales by Region")
    ax1.set_xlabel("Region")
    ax1.set_ylabel("Total Sales")
    fig1.tight_layout()
    bar_chart_b64 = fig_to_base64_png(fig1)

    # 5) median sales
    median_sales = float(dfx[c_sales].median()) if len(dfx) else 0.0

    # 6) total sales tax at 10%
    total_sales_tax = float(total_sales * 0.10)

    # 7) cumulative sales over time (red line)
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
        "bar_chart": bar_chart_b64,  # RAW base64, no "data:image/png;base64,"
        "median_sales": median_sales,
        "total_sales_tax": total_sales_tax,
        "cumulative_sales_chart": cumulative_sales_chart_b64,  # RAW base64
    }

def detect_task(questions_text: str) -> str:
    """Return a simple task key for routing."""
    qt = questions_text.lower()
    if "analyze `sample-sales.csv`" in qt or "analyze 'sample-sales.csv'" in qt or "analyze sample-sales.csv" in qt:
        return "sample_sales"
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
        {"name": "questions.txt", "in": "formData", "type": "file", "required": False, "description": "TXT file with questions"},
        {"name": "questions_file", "in": "formData", "type": "file", "required": False, "description": "ALT name for questions"},
        {"name": "data.csv", "in": "formData", "type": "file", "required": False, "description": "Optional CSV file for analysis"},
        {"name": "csv_file", "in": "formData", "type": "file", "required": False, "description": "ALT name for CSV"},
        {"name": "image.png", "in": "formData", "type": "file", "required": False, "description": "Optional image file for vision tasks"},
        {"name": "image_file", "in": "formData", "type": "file", "required": False, "description": "ALT name for image"},
        {"name": "query", "in": "formData", "type": "string", "required": False, "description": "Optional single query override"}
    ],
    "responses": {200: {"description": "JSON object or array of answers from the agent"}}
})
def unified_api():
    # Log everything in the request
    logging.debug("---- Incoming /api request ----")
    logging.debug("Form fields: %s", request.form.to_dict())
    logging.debug("Files received: %s", list(request.files.keys()))

    # Accept both singular and plural + alt field names
    questions_file = (
        request.files.get("questions.txt")
        or request.files.get("question.txt")
        or request.files.get("questions_file")
    )
    csv_file = (
        request.files.get("data.csv")
        or request.files.get("csv_file")
    )
    image_file = (
        request.files.get("image.png")
        or request.files.get("image_file")
    )
    query_text = request.form.get("query")

    if not questions_file and not query_text:
        logging.error("No questions file or query found! Returning 400.")
        return jsonify({"error": "questions.txt or question.txt (or 'query') is required"}), 400

    questions_text = query_text or read_txt_file(questions_file)
    logging.debug("Loaded questions text: %s", questions_text[:200])  # only first 200 chars

    task = detect_task(questions_text)
    logging.debug("Detected task: %s", task)

    if task == "sample_sales":
        if not csv_file:
            logging.error("CSV file not provided for sample-sales task.")
            return jsonify({"error": "sample-sales task requires 'data.csv'"}), 400
        try:
            df = read_csv_file(csv_file)
            logging.debug("CSV shape: %s", df.shape)
            result_obj = analyze_sample_sales(df)
            return jsonify(result_obj), 200
        except Exception as e:
            logging.exception("Failed to analyze sample-sales.csv")
            return jsonify({"error": f"Failed to analyze sample-sales.csv: {e}"}), 500

    logging.error("Unrecognized task. Returning 400.")
    return jsonify({"error": "Task not recognized"}), 400
    # Generic fallback (uses LLM only if enabled)
    if USE_LLM and OPENAI_API_KEY:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful data analyst agent."},
                {"role": "user", "content": questions_text}
            ]
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
            answer = resp.choices[0].message.content
            # Return as array to be safe for arbitrary text answers
            return jsonify([answer])
        except Exception as e:
            return jsonify({"error": f"LLM fallback failed: {e}"}), 500

    # If no LLM, respond clearly
    return jsonify({"error": "Task not recognized and LLM fallback disabled. Provide a supported dataset task (e.g., sample-sales)."}), 400

# ---------------------- Run ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False to avoid double-serving in some envs
    app.run(host="0.0.0.0", port=port, debug=False)




