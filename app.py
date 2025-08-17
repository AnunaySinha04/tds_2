import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import io
import base64
from flasgger import Swagger, swag_from
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
swagger = Swagger(app)

# ---- Config: set API key & endpoint here ----
OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjEwMDE2MzhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.lF4kTrKjBLHn77hNBZObuUL5BUKx-HWti1XEClwD9CM"  # replace with your actual key
OPENAI_BASE_URL = "https://aipipe.org/openai/v1"
OPENAI_MODEL = "gpt-4o-mini"

# ---- OpenAI client ----
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# In-memory storage for uploaded files
file_storage = {}

def read_csv_file(file_obj):
    try:
        df = pd.read_csv(file_obj)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def read_txt_file(file_obj):
    try:
        content = file_obj.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return content
    except Exception as e:
        print(f"Error reading the text file: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Unified API! Use /api to upload files and query. Swagger docs at /apidocs."

@app.route('/api', methods=['POST'])
@swag_from({
    'tags': ['Unified API'],
    'consumes': ['multipart/form-data'],
    'parameters': [
        {
            'name': 'csv_file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'CSV file to upload'
        },
        {
            'name': 'questions_file',
            'in': 'formData',
            'type': 'file',
            'required': True,
            'description': 'Questions TXT file to upload'
        },
        {
            'name': 'query',
            'in': 'formData',
            'type': 'string',
            'required': False,
            'description': 'User query text (optional if provided in questions file)'
        }
    ],
    'responses': {
        200: {
            'description': 'LLM response with context from uploaded files'
        }
    }
})
def unified_api():
    csv_file = request.files.get('csv_file')
    questions_file = request.files.get('questions_file')
    query_text = request.form.get('query')

    if not csv_file or not questions_file:
        return jsonify({'error': 'Both CSV file and Questions file are required'}), 400

    df = read_csv_file(csv_file)
    if df is not None:
        file_storage['csv'] = df

    questions_content = read_txt_file(questions_file)
    if questions_content:
        file_storage['questions'] = questions_content

    question = query_text if query_text else file_storage.get('questions')
    if not question:
        return jsonify({'error': 'No query provided'}), 400

    result = {}
    try:
        context_parts = []
        if df is not None:
            csv_summary = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": df.head(5).to_dict(orient='records')
            }
            context_parts.append(f"CSV Summary: {csv_summary}")

            try:
                sample_sql = "SELECT COUNT(*) as row_count FROM df"
                query_result = duckdb.query(sample_sql).to_df().to_dict(orient='records')
                context_parts.append(f"DuckDB SQL Result: {query_result}")
            except Exception as sql_err:
                context_parts.append(f"DuckDB Error: {sql_err}")

            try:
                fig, ax = plt.subplots()
                df.head(20).plot(ax=ax)
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                img_b64 = base64.b64encode(img.read()).decode('utf-8')
                plt.close(fig)
                result['plot_base64'] = img_b64
            except Exception as plot_err:
                context_parts.append(f"Plotting Error: {plot_err}")

        full_context = "\n\n".join(context_parts) if context_parts else ""

        messages = []
        if full_context:
            messages.append({"role": "system", "content": f"Here is the user-provided context: {full_context}"})
        messages.append({"role": "user", "content": question})

        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages
        )
        result['llm_answer'] = resp.choices[0].message.content
    except Exception as e:
        result['llm_error'] = str(e)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
