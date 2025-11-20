import os
import pandas as pd
import streamlit as st

from rag_core import (
    load_all_attempts,
    detect_concepts_to_reteach,
    detect_struggling_students,
    load_concept_history,
    SUMMARY_CSV,
)

# ------------------------- إعداد الصفحة العامة ------------------------- #

st.set_page_config(page_title="لوحة المعلم (EduRAG)", layout="wide")
st.title("👨‍🏫 لوحة المعلم")

# ------------------------- دوال مساعدة ------------------------- #

def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _classify_status(row: pd.Series) -> str:
    """تصنيف حالة الطالب بشكل أوضح للمعلم."""
    last_acc = row.get("last_accuracy", 0.0)
    imp = row.get("improvement_pct", 0.0)
    avg_t = row.get("avg_time_sec", 0.0)

    # دعم عاجل
    if last_acc < 60:
        return "يحتاج دعم عاجل"
    # الطالب يتراجع بوضوح
    if imp <= -5:
        return "يتراجع"
    # أداء متوسط أو بطء في الحل
    if last_acc < 75 or avg_t > 90:
        return "يحتاج متابعة"
    # كل شيء جيد
    return "جيد جداً"

# ------------------------- تحميل البيانات الأساسية ------------------------- #

attempts = load_all_attempts()
if attempts.empty:
    st.warning("لا توجد بيانات محاولات حتى الآن. اطلب من الطلاب أداء الاختبارات أولاً.")
    st.stop()

summary_df = _safe_read_csv(SUMMARY_CSV)

# تأكد من وجود الأعمدة المتوقعة في ملخص الطلاب
expected_cols = {"student", "best_accuracy", "last_accuracy", "improvement_pct", "avg_time_sec"}
if not summary_df.empty:
    missing = expected_cols - set(summary_df.columns)
    for col in missing:
        # إذا فيه أعمدة ناقصة، نضيفها بقيم افتراضية حتى لا يتعطل الكود
        summary_df[col] = 0.0

    # إضافة تصنيف واضح لحالة الطالب
    summary_df["teacher_status"] = summary_df.apply(_classify_status, axis=1)

# ------------------------- المفاهيم المقترح إعادة شرحها ------------------------- #

concepts_to_repeat = detect_concepts_to_reteach(threshold_ratio=0.4)

if concepts_to_repeat:
    st.info("🔁 المفاهيم المقترح إعادة شرحها للصف: " + " ، ".join(concepts_to_repeat))
else:
    st.success("لا يوجد حالياً مفهوم معيّن يحتاج إعادة شرح على مستوى أغلب الطلاب.")

st.divider()

# ------------------------- تبويبات لوحة المعلم ------------------------- #

tab_overview, tab_students, tab_one_student, tab_concepts = st.tabs(
    ["📊 نظرة عامة", "🚨 الطلاب الذين يحتاجون دعم", "👤 تحليل طالب محدد", "🧠 تحليل المفاهيم"]
)

# ========================= تبويب: نظرة عامة ========================= #

with tab_overview:
    st.subheader("📊 نظرة عامة على الأداء")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("عدد الطلاب", attempts["student"].nunique())
    with col2:
        st.metric("عدد المحاولات", len(attempts))
    with col3:
        st.metric("متوسط الدقة الكلي", f"{attempts['accuracy'].mean():.1f}٪")

    # متوسط الدرجات حسب الفصول والطلاب
    chap_avg = attempts.groupby("chapter")["accuracy"].mean().reset_index()
    stu_avg = attempts.groupby("student")["accuracy"].mean().reset_index()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### متوسط الدرجات لكل فصل")
        if not chap_avg.empty:
            st.bar_chart(chap_avg.set_index("chapter")["accuracy"])
        else:
            st.info("لا توجد بيانات كافية لعرض متوسط الفصول.")
    with c2:
        st.markdown("### متوسط الدرجات لكل طالب")
        if not stu_avg.empty:
            st.bar_chart(stu_avg.set_index("student")["accuracy"])
        else:
            st.info("لا توجد بيانات كافية لعرض متوسط الطلاب.")

    if not summary_df.empty:
        st.markdown("### ملخص كل الطلاب")
        st.dataframe(
            summary_df[["student", "best_accuracy", "last_accuracy", "improvement_pct", "avg_time_sec", "teacher_status"]]
            .sort_values(["teacher_status", "last_accuracy"], ascending=[True, False])
            .reset_index(drop=True)
        )
    else:
        st.info("لا يوجد ملف ملخص للطلاب حتى الآن (students_summary.csv).")

# ========================= تبويب: الطلاب الذين يحتاجون دعم ========================= #

with tab_students:
    st.subheader("🚨 الطلاب الذين يحتاجون دعم")

    struggling = detect_struggling_students()
    if struggling.empty:
        st.success("لا يوجد حالياً طلاب مصنفون بأنهم يحتاجون دعم إضافي.")
    else:
        # دمج حالة المعلم مع حالة الدالة إن وجد الملخص
        if not summary_df.empty:
            struggling = struggling.merge(
                summary_df[["student", "teacher_status"]],
                on="student",
                how="left"
            )
        st.markdown(f"عدد الطلاب المصنفين (يحتاج دعم): **{len(struggling)}**")
        st.dataframe(
            struggling[
                ["student", "last_accuracy", "improvement_pct", "avg_time_sec"]
                + (["status"] if "status" in struggling.columns else [])
                + (["teacher_status"] if "teacher_status" in struggling.columns else [])
            ].sort_values("last_accuracy")
        )

# ========================= تبويب: تحليل طالب محدد ========================= #

with tab_one_student:
    st.subheader("👤 تحليل طالب محدد")

    students = sorted(attempts["student"].unique().tolist())
    sel_student = st.selectbox("اختر الطالب", students)

    stu_attempts = attempts[attempts["student"] == sel_student].sort_values(["chapter", "attempt"])
    st.markdown("### جميع محاولات الطالب")
    st.dataframe(stu_attempts)

    # ملخص الطالب من ملف students_summary.csv
    if not summary_df.empty:
        ss = summary_df[summary_df["student"] == sel_student]
        if not ss.empty:
            r = ss.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("أفضل دقة", f"{r['best_accuracy']:.1f}٪")
            with c2:
                st.metric("آخر دقة", f"{r['last_accuracy']:.1f}٪")
            with c3:
                st.metric("نسبة التحسن الكلية", f"{r['improvement_pct']:.1f}")
            with c4:
                st.metric("متوسط الزمن لكل محاولة (ثانية)", f"{r['avg_time_sec']:.1f}")

            st.markdown(f"**تصنيف حالة الطالب:** {r.get('teacher_status', 'غير محدد')}")
        else:
            st.info("لا يوجد ملخص محفوظ لهذا الطالب بعد.")
    else:
        st.info("ملف ملخص الطلاب غير متوفر بعد.")

# ========================= تبويب: تحليل المفاهيم ========================= #

with tab_concepts:
    st.subheader("🧠 تحليل المفاهيم على مستوى الصف")

    ch_hist = load_concept_history()
    if ch_hist.empty:
        st.info("لا توجد بيانات مفصلة عن المفاهيم حتى الآن.")
    else:
        # إحصاءات لكل مفهوم: متوسط الدقة وعدد الطلاب المتأثرين
        ch_summary = ch_hist.groupby("concept").agg(
            students_affected=("student", "nunique"),
            avg_accuracy=("accuracy", "mean"),
            attempts_count=("accuracy", "count"),
        ).reset_index()

        ch_summary["avg_accuracy"] = ch_summary["avg_accuracy"].round(2)

        st.markdown("### ملخص المفاهيم (متوسط دقة الطلاب وعدد الطلاب المتأثرين):")
        st.dataframe(ch_summary.sort_values("avg_accuracy"))

        # اختيار مفهوم معيّن
        concept_list = sorted(ch_summary["concept"].unique().tolist())
        sel_concept = st.selectbox("اختر مفهوم لمعرفة الطلاب المتأثرين:", concept_list)

        c_rows = ch_hist[ch_hist["concept"] == sel_concept].copy()
        if not c_rows.empty:
            st.markdown(f"### أداء الطلاب في مفهوم: {sel_concept}")
            c_rows = c_rows.sort_values(["student", "attempt"])
            st.dataframe(c_rows[["student", "chapter", "attempt", "accuracy"]])
        else:
            st.info("لا توجد بيانات لهذا المفهوم بعد.")
