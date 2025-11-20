import time

import pandas as pd
import streamlit as st

from rag_core import (
    load_qna_for_chapter,
    select_first_attempt,
    select_second_attempt,
    grade_attempt,
    get_pages_for_concepts,
    save_attempt,
)

st.set_page_config(page_title="🎓 اختبار الطالب (EduRAG)", layout="centered")
st.title("🎓 اختبار الطالب")

# ---------------------- مدخلات أساسية ---------------------- #

student = st.text_input("🪪 اكتب اسمك هنا").strip()
chapter = st.selectbox("📘 اختر الفصل", [1, 2, 3, 4, 5])

df_qna = load_qna_for_chapter(chapter)
if df_qna.empty:
    st.warning("🚫 لا توجد أسئلة متاحة لهذا الفصل. تأكد من وجود ملفات data/questions_chX.csv و answers_chX.csv.")
    st.stop()

NUM_QUESTIONS = st.slider("عدد الأسئلة في الاختبار", 5, min(20, len(df_qna)), 10)

# حالة الجلسة
if "phase" not in st.session_state:
    st.session_state.phase = "first"
    st.session_state.first_questions = None
    st.session_state.second_questions = None
    st.session_state.first_summary = None
    st.session_state.start_time = None


# ---------------------- التايمر ---------------------- #

def show_timer():
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
    elapsed = int(time.time() - st.session_state.start_time)
    st.markdown(f"⏱️ **الوقت المنقضي:** {elapsed} ثانية", help="يتم حفظ الوقت كميزة (feature) لتحليل أداء الطالب.")
    return elapsed


# ---------------------- عرض الكويز ---------------------- #

def render_quiz(df_questions: pd.DataFrame, key_prefix: str):
    answers = {}
    for i, row in df_questions.iterrows():
        st.markdown(f"**سؤال {i+1}: {row.get('question', '')}**")
        options = []
        for opt in ["a", "b", "c", "d"]:
            col = f"option_{opt}"
            if col in row and pd.notna(row[col]):
                options.append(row[col])
        if not options:
            options = ["لا توجد خيارات"]  # احتياط
        choice = st.radio("اختر الإجابة:", options, key=f"{key_prefix}_{i}")
        answers[int(row["question_id"])] = choice
        st.divider()
    return answers


# ---------------------- المحاولة الأولى ---------------------- #

if st.session_state.phase == "first":
    st.header("🧮 المحاولة الأولى")

    if not student:
        st.info("📝 من فضلك أدخل اسمك للبدء.")
        st.stop()

    if st.session_state.first_questions is None:
        st.session_state.first_questions = select_first_attempt(df_qna, NUM_QUESTIONS)
        st.session_state.start_time = time.time()

    show_timer()
    answers = render_quiz(st.session_state.first_questions, "first")

    if st.button("إرسال الإجابات ✅"):
        res_df, summary = grade_attempt(st.session_state.first_questions, answers)
        time_used = time.time() - st.session_state.start_time

        # حفظ المحاولة + إحصاءات المفاهيم
        save_attempt(
            student=student,
            chapter=chapter,
            attempt_index=1,
            summary=summary,
            time_taken_sec=time_used,
        )

        st.session_state.first_summary = summary

        st.success(f"🎯 نتيجتك: {summary['correct']} من {summary['total']} ({summary['accuracy']:.1f}٪)")

        if summary["weak_concepts"]:
            st.warning("🧩 المفاهيم التي تحتاج مراجعة:")
            for c in summary["weak_concepts"]:
                st.write(f"- {c}")
        else:
            st.info("ممتاز! لم تُسجَّل مفاهيم ضعيفة 👏")

        # صفحات مراجعة من الكتاب
        pages = get_pages_for_concepts(summary["weak_concepts"])
        if pages:
            st.markdown("📚 **ننصحك بمراجعة الصفحات التالية من الكتاب:**")
            for p in pages:
                st.write(f"- الصفحات: {p}")

        # زر المحاولة الثانية
        if st.button("إعادة الكويز مع التركيز على نقاط الضعف 🔁"):
            st.session_state.second_questions = select_second_attempt(
                df_qna,
                NUM_QUESTIONS,
                summary["weak_concepts"],
            )
            st.session_state.phase = "second"
            st.session_state.start_time = time.time()
            st.experimental_rerun()

# ---------------------- المحاولة الثانية ---------------------- #

elif st.session_state.phase == "second":
    st.header("📖 المحاولة الثانية — تركيز على المفاهيم الضعيفة")

    if st.session_state.second_questions is None or st.session_state.second_questions.empty:
        st.info("لا توجد أسئلة متاحة حالياً. أحسنت 👏")
        st.stop()

    show_timer()
    answers2 = render_quiz(st.session_state.second_questions, "second")

    if st.button("إرسال المحاولة الثانية ✅"):
        res2_df, summary2 = grade_attempt(st.session_state.second_questions, answers2)
        time_used2 = time.time() - st.session_state.start_time

        save_attempt(
            student=student,
            chapter=chapter,
            attempt_index=2,
            summary=summary2,
            time_taken_sec=time_used2,
        )

        s1_acc = st.session_state.first_summary["accuracy"] if st.session_state.first_summary else 0.0
        s2_acc = summary2["accuracy"]
        improvement = s2_acc - s1_acc

        st.success(f"🎯 درجتك الآن: {summary2['correct']} من {summary2['total']} ({summary2['accuracy']:.1f}٪)")
        st.info(f"📈 نسبة التحسن في الدقة: {improvement:.1f}٪ 🚀")

        if summary2["weak_concepts"]:
            st.warning("لا تزال توجد مفاهيم تحتاج مراجعة:")
            for c in summary2["weak_concepts"]:
                st.write(f"- {c}")
        else:
            st.success("👏 تم علاج جميع نقاط الضعف تقريباً.")

        pages2 = get_pages_for_concepts(summary2["weak_concepts"])
        if pages2:
            st.markdown("📚 **ننصحك بمراجعة الصفحات التالية من الكتاب:**")
            for p in pages2:
                st.write(f"- الصفحات: {p}")

        st.session_state.phase = "done"

# ---------------------- نهاية الجلسة ---------------------- #

elif st.session_state.phase == "done":
    st.header("🎉 أحسنت!")
    st.write("أنهيت الاختبار والمراجعة بنجاح. يمكنك إغلاق الصفحة أو تجربة فصل آخر.")
