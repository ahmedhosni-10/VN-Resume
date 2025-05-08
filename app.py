# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import textwrap
import pdfplumber

load_dotenv()


app = Flask(__name__)
CORS(app)

# Configure OpenAI client
client = OpenAI(
    base_url=os.getenv("OPENAI_ENDPOINT", "https://models.inference.ai.azure.com"),
    api_key=os.getenv("OPENAI_API_KEY")
)

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def parse_feedback(feedback_text):
    sections = ["Overall Rating", "Summary", "Strengths", "Weaknesses",
                "ATS compatibility analysis", "Formating and readability",
                "Content and impact", "Grammer and clarity"]
    parsed = {}
    current_section = None

    for line in feedback_text.splitlines():
        stripped = line.strip()
        lower_line = stripped.lower()

        # Check for section headers
        for section in sections:
            if lower_line.startswith(section.lower()):
                current_section = section
                parsed[current_section] = ""
                continue

        # Add content to current section
        if current_section and stripped:
            if parsed[current_section]:
                parsed[current_section] += "\n" + stripped
            else:
                parsed[current_section] = stripped

    return parsed

def get_resume_feedback(resume_text):
    prompt = f"""
You are a professional career advisor. Analyze the following resume and provide the following structured feedback:

1. **Overall Rating (out of 100)** — Evaluate the overall quality of the resume.
2. **Summary** — A short paragraph in 1 line summarizing the impression of the resume.
3. **Strengths** — What is working well in this resume? (3-5 bullet points)
4. **Weaknesses** — What is not working well in this resume? (3-5 bullet points)
5. **ATS compatibility analysis** — give an ATS match score for this resume, issue and fix? (each one in one line)
6. **Formating and readability** — what is the issue of formating and fix? (each one in one line)
7. **Content and impact** — what is the issue of content and fix? (each one in one line)
8. **Grammer and clarity** - what is issue of grammer and fix? (each one in one line)
Resume:
{resume_text}
"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful and professional career advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=1.0,
        max_tokens=1000,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    return parse_feedback(response.choices[0].message.content)

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            file.save(temp_file.name)
            resume_text = extract_text_from_pdf(temp_file.name)

        feedback = get_resume_feedback(resume_text)
        return jsonify(feedback)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))