import os
import pickle
import pandas as pd
import numpy as np
import re
import json
import streamlit as st
from sklearn.metrics.pairwise import linear_kernel
from openai import OpenAI

# ----------------------------- إعداد المسارات ----------------------------- #
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
RAG_DIR = os.path.join(BASE_DIR, "rag_data")

# التأكد من وجود المجلدات
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)

ATTEMPTS_CSV = os.path.join(REPORTS_DIR, "attempts.csv")
SUMMARY_CSV = os.path.join(REPORTS_DIR, "students_summary.csv")
CONCEPT_HISTORY_CSV = os.path.join(REPORTS_DIR, "concept_history.csv")

# ----------------------------- 1. دوال RAG والبحث ----------------------------- #

@st.cache_resource
def load_rag_resources():
    try:
        vec_path = os.path.join(RAG_DIR, "vectorizer.pkl")
        mat_path = os.path.join(RAG_DIR, "tfidf_matrix.pkl")
        chk_path = os.path.join(RAG_DIR, "chunks.pkl")
        
        if not (os.path.exists(vec_path) and os.path.exists(mat_path)):
            return None, None, None

        with open(vec_path, 'rb') as f: vectorizer = pickle.load(f)
        with open(mat_path, 'rb') as f: matrix = pickle.load(f)
        with open(chk_path, 'rb') as f: chunks = pickle.load(f)
        return vectorizer, matrix, chunks
    except Exception:
        return None, None, None

def search_concept_in_book(query, top_k=2):
    vectorizer, matrix, chunks = load_rag_resources()
    if not vectorizer: return []

    try:
        query = re.sub(r'[^\w\s]', '', str(query)) 
        query_vec = vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vec, matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_k:-1]
        
        results = []
        for i in related_docs_indices:
            if cosine_similarities[i] > 0.01: 
                results.append(chunks[i])
        return results
    except Exception as e:
        print(f"Search Error: {e}")
        return []

def get_explanation_and_page(api_key, concept):
    context_list = search_concept_in_book(concept)
    if not context_list:
        return "المفهوم غير موجود في الفهرس بدقة.", "-"
    
    pages = sorted(list(set([c['page'] for c in context_list])))
    pages_str = ", ".join(map(str, pages))
    context_text = "\n".join([c['text'] for c in context_list])

    if not api_key:
        return f"راجع الصفحات: {pages_str}\nنص مقتبس: {context_text[:200]}...", pages_str

    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
        اشرح للطالب مفهوم "{concept}" بشكل مبسط جداً (سطرين) بناءً على النص التالي:
        {context_text[:800]}
        """
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return res.choices[0].message.content, pages_str
    except Exception as e:
        return f"خطأ في الاتصال: {e}", pages_str

# ----------------------------- 2. دوال التوليد والتحليل (AI & Analytics) ----------------------------- #

def clean_and_parse_json(content):
    try:
        content = content.replace("```json", "").replace("```", "").strip()
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != -1:
            return json.loads(content[start:end])
        return None
    except:
        return None

def prepare_second_attempt_quiz(api_key, chapter, weak_concepts, total_q=5):
    if not api_key: return pd.DataFrame()

    target_concepts = weak_concepts[:3] if weak_concepts else ["مفاهيم عامة"]
    while len(target_concepts) < total_q:
        target_concepts.append("أسئلة مراجعة عامة")

    client = OpenAI(api_key=api_key)
    prompt = f"""
    Create {total_q} simple math MCQs (Arabic) for Chapter {chapter}.
    Focus on: {', '.join(target_concepts)}.
    OUTPUT FORMAT: JSON Array ONLY.
    Fields: "question", "option_a", "option_b", "option_c", "option_d", "correct_option" (e.g. "option_a"), "concept".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "JSON only."}, {"role": "user", "content": prompt}]
        )
        data = clean_and_parse_json(response.choices[0].message.content)
        if data:
            for q in data: q['question_id'] = f"AI_{np.random.randint(10000, 99999)}"
            return pd.DataFrame(data)
    except: pass
    return pd.DataFrame()

def generate_mixed_quiz(api_key, selected_chapters, num_questions=5):
    """توليد اختبار مركب من عدة فصول"""
    if not api_key: return pd.DataFrame()
    
    # 1. جمع المفاهيم
    all_concepts = []
    for ch in selected_chapters:
        q_file = os.path.join(DATA_DIR, f"questions_ch{ch}.csv")
        if os.path.exists(q_file):
            try:
                df = pd.read_csv(q_file)
                if 'concept' in df.columns:
                    all_concepts.extend(df['concept'].dropna().unique().tolist())
            except: pass
    
    if not all_concepts: return pd.DataFrame()

    chosen_concepts = np.random.choice(all_concepts, min(len(all_concepts), num_questions), replace=False)
    
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Create a mixed math quiz of {num_questions} questions covering Chapters {selected_chapters}.
    Target Concepts: {', '.join(chosen_concepts)}.
    OUTPUT: JSON Array ONLY.
    Fields: "question", "option_a", "option_b", "option_c", "option_d", "correct_option" (e.g. "option_a"), "concept".
    Language: Arabic.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "JSON array output only."}, {"role": "user", "content": prompt}]
        )
        data = clean_and_parse_json(response.choices[0].message.content)
        if data:
            for q in data: q['question_id'] = f"MIX_{np.random.randint(10000,99999)}"
            return pd.DataFrame(data)
    except: pass
    return pd.DataFrame()

def generate_ai_summary(api_key, context_type="general", data=None):
    if not api_key: return "الرجاء إدخال مفتاح API."
    client = OpenAI(api_key=api_key)
    
    if context_type == "general":
        prompt = f"حلل أداء الفصل: متوسط {data.get('avg',0):.1f}%، عدد المتعثرين {data.get('risk_count',0)}. أعط 3 نصائح للمعلم."
    else:
        prompt = "لخص أداء الطالب."
        
    try:
        res = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return res.choices[0].message.content
    except Exception as e: return f"خطأ: {e}"

def detect_concepts_to_reteach(threshold=50):
    if not os.path.exists(CONCEPT_HISTORY_CSV): return pd.DataFrame()
    df = pd.read_csv(CONCEPT_HISTORY_CSV)
    if df.empty: return pd.DataFrame()

    stats = df.groupby('concept').agg(
        total_attempts=('correct', 'count'),
        success_rate=('correct', 'mean')
    ).reset_index()
    stats['success_rate'] = stats['success_rate'] * 100
    return stats[stats['success_rate'] < threshold].sort_values('success_rate')

def get_strict_risk_students():
    if not os.path.exists(SUMMARY_CSV): return pd.DataFrame()
    df = pd.read_csv(SUMMARY_CSV)
    risk_list = []
    for _, row in df.iterrows():
        reasons = []
        score = row.get('last_accuracy', 0)
        imp = row.get('improvement_pct', 0)
        if score < 50: reasons.append("مستوى حرج")
        elif score < 65 and imp < 0: reasons.append("تراجع مستمر")
        
        if reasons:
            risk_list.append({
                "الطالب": row['student'],
                "درجة الخطورة": "عالية",
                "الأسباب": ", ".join(reasons),
                "آخر درجة": score
            })
    return pd.DataFrame(risk_list)

# ----------------------------- 3. دوال تحميل البيانات (بما فيها الدالة المفقودة) ----------------------------- #

def load_concept_history():
    """الدالة التي كانت مفقودة وتسبب الخطأ"""
    if os.path.exists(CONCEPT_HISTORY_CSV):
        return pd.read_csv(CONCEPT_HISTORY_CSV)
    return pd.DataFrame()

def load_all_data():
    sum_df = pd.read_csv(SUMMARY_CSV) if os.path.exists(SUMMARY_CSV) else pd.DataFrame()
    att_df = pd.read_csv(ATTEMPTS_CSV) if os.path.exists(ATTEMPTS_CSV) else pd.DataFrame()
    con_df = load_concept_history() # استخدام الدالة لضمان الاتساق
    return sum_df, att_df, con_df

def load_qna_for_chapter(chapter):
    try:
        q_file = os.path.join(DATA_DIR, f"questions_ch{chapter}.csv")
        a_file = os.path.join(DATA_DIR, f"answers_ch{chapter}.csv")
        if os.path.exists(q_file) and os.path.exists(a_file):
            q = pd.read_csv(q_file)
            a = pd.read_csv(a_file)
            if 'correct_option' in q.columns: q = q.drop(columns=['correct_option'])
            q['question_id'] = q['question_id'].astype(str)
            a['question_id'] = a['question_id'].astype(str)
            return pd.merge(q, a, on="question_id", how="inner")
    except: pass
    return pd.DataFrame()

# ----------------------------- 4. دوال التصحيح والحفظ ----------------------------- #

def grade_attempt(questions, user_answers):
    correct = 0
    weak_concepts = []
    details = []
    
    if 'correct_option' not in questions.columns:
        return {"total": 0, "correct": 0, "accuracy": 0, "weak_concepts": [], "details": []}

    for _, row in questions.iterrows():
        qid = str(row['question_id'])
        ua = str(user_answers.get(qid, "None")).strip()
        ca = str(row['correct_option']).strip()
        correct_text = str(row.get(ca, ca)) # للحصول على النص
        
        is_correct = (ua == ca)
        if not is_correct and str(row.get(ca, "")).strip() == ua: is_correct = True
            
        if is_correct: correct += 1
        else: weak_concepts.append(row['concept'])
        
        details.append({
            "question": row['question'],
            "user_ans": ua,
            "correct_ans": correct_text,
            "is_correct": is_correct,
            "concept": row['concept']
        })

    return {
        "total": len(questions),
        "correct": correct,
        "accuracy": (correct/len(questions))*100 if len(questions) > 0 else 0,
        "weak_concepts": list(set(weak_concepts)),
        "details": details
    }

def save_attempt_data(student, chapter, attempt, summary, time_sec):
    row = {
        "student": student, "chapter": chapter, "attempt": attempt,
        "total": summary['total'], "correct": summary['correct'],
        "accuracy": summary['accuracy'], "weak_concepts": ";".join(summary['weak_concepts']),
        "time_sec": time_sec
    }
    df = pd.DataFrame([row])
    hdr = not os.path.exists(ATTEMPTS_CSV)
    df.to_csv(ATTEMPTS_CSV, mode='a', header=hdr, index=False)
    update_student_summary(student)
    
    c_rows = []
    for d in summary['details']:
        c_rows.append({
            "student": student, "chapter": chapter, "attempt": attempt,
            "concept": d['concept'], "correct": 1 if d['is_correct'] else 0,
            "total": 1, "accuracy": 100 if d['is_correct'] else 0
        })
    if c_rows:
        c_df = pd.DataFrame(c_rows)
        c_hdr = not os.path.exists(CONCEPT_HISTORY_CSV)
        c_df.to_csv(CONCEPT_HISTORY_CSV, mode='a', header=c_hdr, index=False)

def update_student_summary(student):
    if not os.path.exists(ATTEMPTS_CSV): return
    df = pd.read_csv(ATTEMPTS_CSV)
    s_df = df[df['student'] == student]
    if s_df.empty: return
    
    summ_row = {
        "student": student, 
        "best_accuracy": s_df['accuracy'].max(), 
        "last_accuracy": s_df.iloc[-1]['accuracy'], 
        "improvement_pct": s_df.iloc[-1]['accuracy'] - s_df.iloc[0]['accuracy'], 
        "avg_time_sec": s_df['time_sec'].mean()
    }
    
    full_sum = pd.read_csv(SUMMARY_CSV) if os.path.exists(SUMMARY_CSV) else pd.DataFrame()
    if not full_sum.empty: full_sum = full_sum[full_sum['student'] != student]
    pd.concat([full_sum, pd.DataFrame([summ_row])]).to_csv(SUMMARY_CSV, index=False)