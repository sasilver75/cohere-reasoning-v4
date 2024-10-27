import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from flask.json import provider

app = Flask(__name__)

# Load the CSV file
csv_path = "datasets/cn_k12_math_problems_completions_command-r-plus-08-2024_2_5_1_ON.csv"
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)


@app.route("/")
def index():
    page = request.args.get("page", 1, type=int)
    if page < 1 or page > len(df):
        page = 1

    row = df.iloc[page - 1]
    completion_data = {
        "row_id": int(row.get("row_id", 0)),
        "solution_idx": int(row.get("solution_idx", 0)),
        "problem": str(row.get("problem", "N/A")),
        "ground_truth_solution": str(row.get("ground_truth_solution", "N/A")),
        "candidate_solution": str(row.get("candidate_solution", "N/A")),
        "verification_reasoning": str(row.get("verification_reasoning", "N/A")),
        "verification": bool(row.get("verification", False)),
        "prefix_reasoning": str(row.get("prefix_reasoning", "N/A")),
        "prefix": str(row.get("prefix", "N/A")),
        "completion": str(row.get("completion", "N/A")),
        "completion_verification_reasoning": str(
            row.get("completion_verification_reasoning", "N/A")
        ),
        "completion_verification": bool(row.get("completion_verification", False)),
        "completion_prefix_reasoning": str(
            row.get("completion_prefix_reasoning", "N/A")
        ),
        "completion_verification_prefix": str(
            row.get("completion_verification_prefix", "N/A")
        ),
    }
    # Create a new field for full completion
    completion_data["full_completion"] = (
        completion_data["prefix"] + " " + completion_data["completion"]
    )

    return render_template_string(
        """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Completions Viewer</title>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <script>
            MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\(', '\\)']],
                    displayMath: [['$$', '$$'], ['\\[', '\\]']],
                    processEscapes: true,
                    processEnvironments: true
                },
                options: {
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
                }
            };
        </script>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                padding: 20px;
                max-width: 1800px;
                margin: 0 auto;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .completion { 
                border: 1px solid #ddd; 
                padding: 20px; 
                margin-bottom: 20px;
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }
            .section {
                width: 48%;
                margin-bottom: 20px;
            }
            h2, h3 { color: #333; }
            .math-content, .verification-content { 
                background-color: #f4f4f4; 
                padding: 10px; 
                word-wrap: break-word;
                overflow-wrap: break-word;
                white-space: normal;
                margin-bottom: 10px;
            }
            .verification-content {
                background-color: {{ 'lightgreen' if completion_data.verification else 'lightcoral' }};
            }
            .completion-verification-content {
                background-color: {{ 'lightgreen' if completion_data.completion_verification else 'lightcoral' }};
            }
            .navigation { 
                display: flex; 
                gap: 10px;
            }
            .navigation a { 
                text-decoration: none; 
                color: #333; 
                padding: 10px; 
                border: 1px solid #ddd;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Completions Viewer ({{ page }}/{{ total_pages }})</h1>
            <div class="navigation">
                {% if page > 1 %}
                    <a href="{{ url_for('index', page=page-1) }}">Previous</a>
                {% else %}
                    <span></span>
                {% endif %}
                {% if page < total_pages %}
                    <a href="{{ url_for('index', page=page+1) }}">Next</a>
                {% else %}
                    <span></span>
                {% endif %}
            </div>
        </div>
        <h2>Row ID: {{ completion_data.row_id }} | Solution Index: {{ completion_data.solution_idx }}</h2>
        <div class="completion">
            <div class="section">
                <h2>Problem:</h2>
                <div class="math-content">{{ completion_data.problem }}</div>
                
                <h2>Ground Truth Solution:</h2>
                <div class="math-content">{{ completion_data.ground_truth_solution }}</div>
                
                <hr> <!-- Horizontal divider -->
                
                <h2>Candidate Solution:</h2>
                <div class="math-content">{{ completion_data.candidate_solution }}</div>
                
                <h2>Candidate Solution Verification Reasoning:</h2>
                <div class="math-content">{{ completion_data.verification_reasoning }}</div>
                
                <h2>Candidate Solution Verification:</h2>
                <div class="verification-content">{{ completion_data.verification }}</div>
                
                <h2>Candidate Solution Prefix:</h2>
                <div class="math-content">{{ completion_data.prefix }}</div>
            </div>
            <div class="section">
                <h2>Candidate Solution Prefix:</h2>
                <div class="math-content">{{ completion_data.prefix }}</div>

                <h2>Completion:</h2>
                <div class="math-content">{{ completion_data.completion }}</div>

                <hr> <!-- Horizontal divider -->
                
                <h2>Full Completion (Prefix + Completion):</h2>
                <div class="math-content">{{ completion_data.full_completion }}</div>
                
                <h2>Completion Verification Reasoning:</h2>
                <div class="math-content">{{ completion_data.completion_verification_reasoning }}</div>
                
                <h2>Completion Verification:</h2>
                <div class="completion-verification-content">{{ completion_data.completion_verification }}</div>
            </div>
        </div>
    </body>
    </html>
    """,
        completion_data=completion_data,
        page=page,
        total_pages=len(df),
    )


# Custom JSON encoder to handle numpy types
class NumpyEncoder(provider.DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json = NumpyEncoder(app)

if __name__ == "__main__":
    print(f"Starting server. CSV file path: {csv_path}")
    app.run(debug=True, host="localhost", port=5000)
