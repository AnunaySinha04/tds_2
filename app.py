import os
import io
import base64
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from flasgger import Swagger, swag_from
from openai import OpenAI

# ---------------------- Flask Setup ----------------------
app = Flask(__name__)
swagger = Swagger(app)

# ---------------------- Config ----------------------
OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjEwMDE2MzhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.lF4kTrKjBLHn77hNBZObuUL5BUKx-HWti1XEClwD9CM"  # replace with your actual key

OPENAI_BASE_URL = "https://aipipe.org/openai/v1"
OPENAI_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# In-memory storage
file_storage = {}

# ---------------------- Helper Functions ----------------------
def read_csv_file(file_obj):
    try:
        return pd.read_csv(file_obj)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def read_txt_file(file_obj):
    try:
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content.strip()
    except Exception as e:
        print(f"Error reading TXT: {e}")
        return None

def encode_image(file_obj):
    try:
        return base64.b64encode(file_obj.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def generate_csv_plot(df):
    try:
        fig, ax = plt.subplots()
        df.head(20).plot(ax=ax)
        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches='tight')
        plt.close(fig)
        img.seek(0)
        img_b64 = base64.b64encode(img.read()).decode("utf-8")
        return f"data:image/png;base64,{img_b64}"
    except Exception as e:
        print(f"Error plotting CSV: {e}")
        return None

# ---------------------- Routes ----------------------
@app.route("/", methods=["GET"])
def index():
    return "Welcome! Use /api for file upload & querying. Swagger docs at /apidocs"

@app.route("/api", methods=["POST"])
@swag_from({
    "tags": ["Data Analyst Agent API"],
    "consumes": ["multipart/form-data"],
    "parameters": [
        {"name": "questions_file", "in": "formData", "type": "file", "required": True, "description": "TXT file with questions"},
        {"name": "csv_file", "in": "formData", "type": "file", "required": False, "description": "Optional CSV file for analysis"},
        {"name": "image_file", "in": "formData", "type": "file", "required": False, "description": "Optional image file for vision tasks"},
        {"name": "query", "in": "formData", "type": "string", "required": False, "description": "Optional single query"}
    ],
    "responses": {200: {"description": "JSON array of answers from LLM"}}
})
def unified_api():
    questions_file = request.files.get("questions_file")
    csv_file = request.files.get("csv_file")
    image_file = request.files.get("image_file")
    query_text = request.form.get("query")

    if not questions_file:
        return jsonify({"error": "questions_file is required"}), 400

    # Read questions
    questions_content = read_txt_file(questions_file)
    questions_list = [q.strip() for q in questions_content.split("\n") if q.strip()]

    # Override with single query if provided
    if query_text:
        questions_list = [query_text]

    result = []

    # Prepare context
    context_parts = []

    # CSV context
    df = None
    if csv_file:
        df = read_csv_file(csv_file)
        if df is not None:
            csv_summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": df.head(5).to_dict(orient="records")
            }
            context_parts.append(f"CSV Summary: {csv_summary}")

            # DuckDB SQL example
            try:
                query_result = duckdb.query("SELECT COUNT(*) AS row_count FROM df").to_df().to_dict(orient="records")
                context_parts.append(f"DuckDB SQL Result: {query_result}")
            except Exception as sql_err:
                context_parts.append(f"DuckDB Error: {sql_err}")

            # CSV plot
            csv_plot_b64 = generate_csv_plot(df)
            if csv_plot_b64:
                context_parts.append(f"CSV Plot: {csv_plot_b64}")

    # Image context
    if image_file:
        img_b64 = encode_image(image_file)
        if img_b64:
            context_parts.append(f"Image provided: data:image/png;base64,{img_b64}")

    # Full system context for LLM
    full_context = "\n\n".join(context_parts) if context_parts else "No additional context."

    # Process each question
    for q in questions_list:
        try:
            messages = [
                {"role": "system", "content": f"You are a Data Analyst Agent. Use the context to answer questions:\n{full_context}"},
                {"role": "user", "content": q}
            ]
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
            answer = resp.choices[0].message.content
            result.append(answer)
        except Exception as e:
            result.append(f"Error processing question: {str(e)}")

    return jsonify(result)

# ---------------------- Run ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
