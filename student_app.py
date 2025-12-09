import time
import pandas as pd
import streamlit as st
from rag_core import (
    load_qna_for_chapter,
    grade_attempt,
    save_attempt_data,
    get_explanation_and_page,
    prepare_second_attempt_quiz
)

# إعداد الصفحة بعنوان رسمي وتصميم بسيط
st.set_page_config(page_title="EduRAG - نظام التقييم الأكاديمي", layout="centered")

# CSS لإخفاء القوائم غير الضرورية وجعل الخطوط أكثر رسمية
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #f0f2f6;
        color: #31333F;
        border: 1px solid #d6d6d6;
    }
    .stButton>button:hover {
        border-color: #0056b3;
        color: #0056b3;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- إدارة الحالة (Session State) ------------------- #
if 'step' not in st.session_state: st.session_state.step = 'login'
if 'student_name' not in st.session_state: st.session_state.student_name = ""
if 'api_key' not in st.session_state: st.session_state.api_key = ""

# ------------------- الشريط الجانبي (إعدادات النظام) ------------------- #
with st.sidebar:
    st.header("إعدادات النظام")
    api_input = st.text_input("مفتاح واجهة التطبيقات (API Key)", type="password")
    if api_input: st.session_state.api_key = api_input
    
    st.markdown("---")
    if st.button("تسجيل الخروج"):
        st.session_state.clear()
        st.rerun()

# ------------------- 1. تسجيل الدخول ------------------- #
if st.session_state.step == 'login':
    st.title("نظام التقييم الذكي")
    st.markdown("يرجى إدخال بيانات الطالب للمتابعة.")
    
    name = st.text_input("اسم الطالب الرباعي:")
    if st.button("تسجيل الدخول") and name:
        st.session_state.student_name = name
        st.session_state.step = 'select_chapter'
        st.rerun()

# ------------------- 2. اختيار المقرر ------------------- #
elif st.session_state.step == 'select_chapter':
    st.title(f"لوحة الطالب: {st.session_state.student_name}")
    st.subheader("اختيار وحدة التقييم")
    
    chapter = st.selectbox("اختر الفصل الدراسي:", [1, 2, 3, 4, 5])
    
    if st.button("بدء التقييم الأساسي"):
        df = load_qna_for_chapter(chapter)
        if not df.empty:
            st.session_state.chapter = chapter
            st.session_state.questions = df.sample(5).reset_index(drop=True)
            st.session_state.attempt_num = 1
            st.session_state.start_time = time.time()
            st.session_state.step = 'quiz'
            st.rerun()
        else:
            st.error("خطأ في النظام: لم يتم العثور على بنك الأسئلة لهذا الفصل.")

# ------------------- 3. واجهة الاختبار ------------------- #
elif st.session_state.step == 'quiz':
    exam_type = "الأساسي" if st.session_state.attempt_num == 1 else "التعويضي (مكيف)"
    st.subheader(f"نموذج الاختبار: الفصل {st.session_state.chapter} - {exam_type}")
    
    questions = st.session_state.questions
    user_answers = {}
    
    with st.form("quiz_form"):
        for idx, row in questions.iterrows():
            st.markdown(f"**سؤال {idx+1}:** {row['question']}")
            ops = [row.get('option_a'), row.get('option_b'), row.get('option_c'), row.get('option_d')]
            ops = [str(o) for o in ops if pd.notna(o)]
            
            user_answers[row['question_id']] = st.radio(
                "اختر الإجابة الصحيحة:", ops, key=row['question_id'], index=None
            )
            st.markdown("---")
            
        if st.form_submit_button("اعتماد وإرسال الإجابات"):
            summary = grade_attempt(questions, user_answers)
            save_attempt_data(
                st.session_state.student_name,
                st.session_state.chapter,
                st.session_state.attempt_num,
                summary,
                time.time() - st.session_state.start_time
            )
            st.session_state.last_summary = summary
            st.session_state.step = 'results'
            st.rerun()

# ------------------- 4. تقرير النتائج ------------------- #
elif st.session_state.step == 'results':
    summary = st.session_state.last_summary
    st.title("تقرير الأداء الأكاديمي")
    
    # بطاقة الدرجات
    col1, col2, col3 = st.columns(3)
    col1.metric("الدرجة النهائية", f"{summary['accuracy']:.1f}%")
    col2.metric("الإجابات الصحيحة", f"{summary['correct']} / {summary['total']}")
    col3.metric("الحالة", "ناجح" if summary['accuracy'] >= 60 else "يحتاج تحسين")
    
    st.markdown("### التحليل التفصيلي للإجابات")
    
    for detail in summary['details']:
        status_color = "green" if detail['is_correct'] else "red"
        with st.expander(f"سؤال: {detail['concept']}", expanded=not detail['is_correct']):
            st.markdown(f"**نص السؤال:** {detail['question']}")
            st.markdown(f"**إجابة الطالب:** {detail['user_ans']}")
            st.markdown(f"**الإجابة النموذجية:** :green[{detail['correct_ans']}]")
            
            if not detail['is_correct']:
                st.markdown("---")
                st.markdown("**التوجيه الأكاديمي (AI):**")
                explanation, pages = get_explanation_and_page(st.session_state.api_key, detail['concept'])
                st.info(f"المرجع المنهجي: صفحة {pages}")
                st.write(explanation)

    if st.session_state.attempt_num == 1:
        st.markdown("---")
        st.subheader("الخيارات المتاحة")
        st.info("يمكنك إجراء اختبار تعويضي يركز على نقاط الضعف المحددة في التقرير أعلاه.")
        
        if st.button("بدء الاختبار التعويضي"):
            if not st.session_state.api_key:
                st.error("يتطلب الاختبار التعويضي مفتاح API نشط.")
            else:
                with st.spinner("جاري إعداد نموذج اختبار مخصص..."):
                    new_quiz = prepare_second_attempt_quiz(
                        st.session_state.api_key,
                        st.session_state.chapter,
                        summary['weak_concepts']
                    )
                    
                    if not new_quiz.empty:
                        st.session_state.questions = new_quiz
                        st.session_state.attempt_num = 2
                        st.session_state.start_time = time.time()
                        st.session_state.step = 'quiz'
                        st.rerun()
                    else:
                        st.error("تعذر إنشاء الاختبار في الوقت الحالي. يرجى المحاولة لاحقاً.")
    
    if st.button("العودة للصفحة الرئيسية"):
        st.session_state.step = 'select_chapter'
        st.rerun()