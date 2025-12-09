
# EduRAG_Pro — Advanced Educational RAG System

EduRAG_Pro is an AI‑powered educational platform that integrates Retrieval‑Augmented Generation (RAG), interactive assessments, and performance analytics to support both students and teachers. The system provides instant feedback, weakness detection, and personalized study recommendations based on the student’s answers.

---

## Key Features
- **Student Interface:** Dynamic quizzes, immediate scoring, and page recommendations.
- **Teacher Dashboard:** Class analytics, performance statistics, and difficulty heatmaps.
- **RAG Integration:** Retrieves relevant explanations from educational PDFs.
- **Structured Dataset:** Organized question/answer CSVs for each chapter.
- **Modular Architecture:** Easy to extend, customize, or integrate into existing systems.

---

## Project Structure
```
EduRAG_Pro/
│── student_app.py
│── teacher_app.py
│── analyzer.py
│── generate_questions_BaCkuP.py
│── requirements.txt
│── README.md
│── data/
│   ├── questions_ch*.csv
│   ├── answers_ch*.csv
│── rag/
│── reports/
│── math.pdf
```

---

## How to Run

### 1) Create Virtual Environment
```
python3 -m venv venv
source venv/bin/activate
```

### 2) Install Dependencies
```
pip install -r requirements.txt
```

### 3) Launch Student Dashboard
```
streamlit run student_app.py
```

### 4) Launch Teacher Dashboard
```
streamlit run teacher_app.py
```

---

## ML + RAG Workflow
- **ML Model:** Linear Regression (expandable to SVM, RandomForest, MLP).
- **RAG:** Extracts indexed content from PDFs (math.pdf) and links the student’s wrong answers to relevant pages.

---

## Developer
**Abdulaziz Alshaer**  
Artificial Intelligence — University of Hail  
GitHub: https://github.com/abdulaziz1811
