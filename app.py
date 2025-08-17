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

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

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

def encode_image_file(file_obj):
    try:
        return base64.b64encode(file_obj.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# ---------------------- Routes ----------------------
@app.route("/", methods=["GET"])
def index():
    return "Welcome! Use /api for file upload & querying. Swagger docs at /apidocs"

@app.route("/api", methods=["POST"])
@swag_from({
    "tags": ["Unified API"],
    "consumes": ["multipart/form-data"],
    "parameters": [
        {
            "name": "csv_file",
            "in": "formData",
            "type": "file",
            "required": True,
            "description": "Upload CSV file"
        },
        {
            "name": "questions_file",
            "in": "formData",
            "type": "file",
            "required": True,
            "description": "Upload TXT file containing questions"
        },
        {
            "name": "image_file",
            "in": "formData",
            "type": "file",
            "required": False,
            "description": "Optional image file (png/jpg)"
        },
        {
            "name": "query",
            "in": "formData",
            "type": "string",
            "required": False,
            "description": "User query (overrides questions.txt if provided)"
        }
    ],
    "responses": {
        200: {"description": "Response from LLM with context from CSV, TXT, and optional image"}
    }
})
def unified_api():
    print("Files received:", request.files)
    print("Form received:", request.form)

    csv_file = request.files.get("csv_file")
    questions_file = request.files.get("questions_file")
    query_text = request.form.get("query")

    if not csv_file or not questions_file:
        return jsonify({
            "error": "Both CSV and TXT files are required",
            "received_files": list(request.files.keys()),
            "received_form": request.form.to_dict()
        }), 400

    # csv_file = request.files.get("csv_file")
    # questions_file = request.files.get("questions_file")
    image_file = request.files.get("image_file")
    # query_text = request.form.get("query")

    # if not csv_file or not questions_file:
    #     return jsonify({"error": "Both CSV and TXT files are required"}), 400

    # Save CSV
    df = read_csv_file(csv_file)
    if df is not None:
        file_storage["csv"] = df

    # Save questions
    questions_content = read_txt_file(questions_file)
    if questions_content:
        file_storage["questions"] = questions_content

    # Save image (if provided)
    if image_file:
        encoded_img = encode_image_file(image_file)
        if encoded_img:
            file_storage["image"] = encoded_img

    # Decide the actual query
    question = query_text if query_text else file_storage.get("questions")
    if not question:
        return jsonify({"error": "No query provided"}), 400

    result = {}
    try:
        context_parts = []

        # CSV summary
        if df is not None:
            csv_summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": df.head(5).to_dict(orient="records")
            }
            context_parts.append(f"CSV Summary: {csv_summary}")

            # Run SQL with DuckDB
            try:
                query_result = duckdb.query("SELECT COUNT(*) AS row_count FROM df").to_df().to_dict(orient="records")
                context_parts.append(f"DuckDB SQL Result: {query_result}")
            except Exception as sql_err:
                context_parts.append(f"DuckDB Error: {sql_err}")

            # Plotting
            try:
                fig, ax = plt.subplots()
                df.head(20).plot(ax=ax)
                img = io.BytesIO()
                plt.savefig(img, format="png")
                img.seek(0)
                result["plot_base64"] = base64.b64encode(img.read()).decode("utf-8")
                plt.close(fig)
            except Exception as plot_err:
                context_parts.append(f"Plotting Error: {plot_err}")

        # Build LLM messages
        messages = []
        if context_parts:
            messages.append({"role": "system", "content": "\n\n".join(context_parts)})
        if "image" in file_storage:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{file_storage['image']}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": question})

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        result["llm_answer"] = resp.choices[0].message.content

    except Exception as e:
        result["llm_error"] = str(e)

    return jsonify(result)

# ---------------------- Run ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
