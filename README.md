# EduRAG Pro: AI-Driven Adaptive Learning & Analytics System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/AI-GPT--3.5%2F4-green)](https://openai.com/)
[![RAG](https://img.shields.io/badge/Technique-RAG-orange)](https://arxiv.org/abs/2005.11401)

## 📖 Executive Summary
**EduRAG Pro** is an advanced educational platform designed to bridge the gap between assessment and personalized remediation. Leveraging **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**, the system provides a dual-interface solution: 
1. An **Adaptive Student Portal** that offers immediate, context-aware feedback derived directly from curricular textbooks.
2. A **Pedagogical Teacher Dashboard** equipped with rule-based heuristics to identify at-risk students and generate psychometrically balanced assessments.

---

## 🏗️ System Architecture & Key Features

### 1. The Student Module (Adaptive Learning)
* **Two-Tier Assessment Logic:**
    * *Phase 1 (Diagnostic):* Standardized questions from a structured question bank.
    * *Phase 2 (Remedial):* AI-generated questions dynamically created to target specific knowledge gaps identified in Phase 1.
* **RAG-Powered AI Tutor:** Utilizes **TF-IDF Vectorization** and **Cosine Similarity** to retrieve precise excerpts from the uploaded textbook (`math.pdf`) and generate simplified explanations using GPT models.
* **Academic Report Card:** Instant visualization of performance metrics and concept mastery.

### 2. The Teacher Module (Decision Support System)
* **Heuristic Risk Analysis:** Implements strict, rule-based algorithms to flag students exhibiting specific failure patterns (e.g., scores < 50% or negative improvement trends).
* **Curriculum Analytics:**
    * *Difficulty Matrix:* Visualizes concept difficulty to inform reteaching strategies.
    * *Grade Distribution:* Histogram analysis to monitor class performance normality.
* **Mixed-Domain Exam Generator:** An AI tool capable of synthesizing comprehensive exams covering multiple chapters, ensuring curricular alignment via RAG context injection.

---

## 🛠️ Technical Stack

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Interactive, responsive UI for students and teachers. |
| **Core Logic** | Python | Backend processing and state management. |
| **AI & NLP** | OpenAI API (GPT) | Text generation, explanation, and question synthesis. |
| **Information Retrieval** | Scikit-Learn (TF-IDF) | Vector space modeling for searching the textbook. |
| **Data Visualization** | Altair | Interactive academic charting and heatmaps. |
| **Document Processing** | PyMuPDF (Fitz) | Extracting and chunking text from PDFs. |

---

## 🚀 Installation & Setup Guide

### Prerequisites
* Python 3.8 or higher.
* An active OpenAI API Key.

### Step 1: Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/EduRAG_Pro.git](https://github.com/YOUR_USERNAME/EduRAG_Pro.git)
cd EduRAG_Pro
Step 2: Configure Environment

It is recommended to use a virtual environment:

Bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Step 3: Install Dependencies

Bash
pip install -r requirements.txt
Step 4: Initialize Knowledge Base (Critical)

Run the indexer to process the textbook (math.pdf) and build the vector search index:

Bash
python build_index.py
Output: rag_data/ directory containing vectorized chunks.

Step 5: Launch Applications

Student Portal:

Bash
streamlit run student_app.py
Teacher Dashboard:

Bash
streamlit run teacher_app.py
📂 Project Structure
Plaintext
EduRAG_Pro/
├── student_app.py          # Student interface entry point
├── teacher_app.py          # Teacher dashboard entry point
├── rag_core.py             # Core engine (RAG logic, grading, AI calls)
├── build_index.py          # PDF indexing script (TF-IDF)
├── math.pdf                # Source curriculum document
├── requirements.txt        # Project dependencies
├── data/                   # Structured Question Bank (CSV)
│   ├── questions_ch1.csv
│   └── answers_ch1.csv
├── reports/                # Generated analytics (Attempts & History)
└── rag_data/               # Serialized Vector Indices (.pkl)
🔒 Security & Privacy
API Key Safety: API keys are never hardcoded. They are input via the secure sidebar session and are not stored persistently on the server.

Data Integrity: Student records are maintained in local structured CSV files for easy auditing and export.

📜 License
This project is intended for educational and research purposes.

Developer: Abdulaziz Abdulrahman 
