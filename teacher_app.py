import streamlit as st
import pandas as pd
import altair as alt
from rag_core import (
    load_all_data, 
    detect_concepts_to_reteach, 
    get_strict_risk_students, 
    generate_ai_summary,
    generate_mixed_quiz,
    load_concept_history # تأكد من وجود دالة تحميل المفاهيم
)

st.set_page_config(page_title="بوابة المعلم - EduRAG Pro", layout="wide")

# CSS لتنسيق الجداول والعناوين بشكل رسمي
st.markdown("""
<style>
    h1, h2, h3 { font-family: 'Arial', sans-serif; color: #2C3E50; }
    .metric-box { border: 1px solid #ddd; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ---------------------- الشريط الجانبي ---------------------- #
with st.sidebar:
    st.header("لوحة التحكم")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    selected_chapters_filter = st.multiselect(
        "نطاق التحليل (الفصول):", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5]
    )

# تحميل البيانات
sum_df, att_df, con_df = load_all_data()

st.title("لوحة المعلومات الأكاديمية")

# ---------------------- مؤشرات الأداء الرئيسية (KPIs) ---------------------- #
if sum_df.empty:
    st.warning("لا توجد بيانات طلاب متاحة للعرض.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
avg_score = sum_df['last_accuracy'].mean()
risk_count = len(get_strict_risk_students())

col1.metric("إجمالي الطلاب", len(sum_df))
col2.metric("متوسط الأداء العام", f"{avg_score:.1f}%")
col3.metric("الطلاب تحت الملاحظة", risk_count, delta=-risk_count, delta_color="inverse")
col4.metric("معدل التحسن", f"{sum_df['improvement_pct'].mean():.1f}%")

st.markdown("---")

# ---------------------- التحليل البياني المتقدم ---------------------- #
tab_overview, tab_concepts, tab_students, tab_exam = st.tabs([
    "نظرة عامة وتوزيع الدرجات", 
    "تحليل المفاهيم والمناهج", 
    "سجلات الطلاب", 
    "بناء الاختبارات"
])

# 1. تبويب نظرة عامة (Distribution Logic)
with tab_overview:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("توزيع الدرجات (Histogram)")
        # منطق الرسم: تقسيم الدرجات إلى فئات (Bins) لرؤية التوزيع الطبيعي
        # هذا الرسم منطقي جداً للمعلم لمعرفة مستوى صعوبة الاختبار
        base = alt.Chart(sum_df).encode(x=alt.X('last_accuracy', bin=alt.Bin(maxbins=10), title='نطاق الدرجات'))
        chart = base.mark_bar(color='#3498db').encode(
            y=alt.Y('count()', title='عدد الطلاب'),
            tooltip=['count()']
        ).properties(height=350)
        st.altair_chart(chart, use_container_width=True)
        st.caption("يوضح الرسم أعلاه تركز درجات الطلاب. الانحياز لليمين يعني سهولة الاختبار، ولليسار يعني صعوبته.")

    with c2:
        st.subheader("تحليل المخاطر")
        risk_df = get_strict_risk_students()
        if not risk_df.empty:
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        else:
            st.success("جميع الطلاب في النطاق الآمن.")

# 2. تبويب تحليل المفاهيم (Difficulty Logic)
with tab_concepts:
    st.subheader("مصفوفة صعوبة المفاهيم")
    
    if not con_df.empty:
        # تجميع البيانات لحساب نسبة النجاح لكل مفهوم
        concept_stats = con_df.groupby('concept')['correct'].mean().reset_index()
        concept_stats['accuracy'] = concept_stats['correct'] * 100
        concept_stats = concept_stats.sort_values('accuracy') # الأقل دقة أولاً (الأصعب)

        # رسم بياني شريطي أفقي (أفضل لقراءة أسماء المفاهيم الطويلة)
        chart_concepts = alt.Chart(concept_stats).mark_bar().encode(
            x=alt.X('accuracy', title='متوسط نسبة الإتقان %', scale=alt.Scale(domain=[0, 100])),
            y=alt.Y('concept', sort='x', title='المفهوم العلمي'),
            color=alt.condition(
                alt.datum.accuracy < 50,
                alt.value('#e74c3c'),  # أحمر للمفاهيم الصعبة
                alt.value('#2ecc71')   # أخضر للمفاهيم السهلة
            ),
            tooltip=['concept', alt.Tooltip('accuracy', format='.1f')]
        ).properties(height=400)
        
        st.altair_chart(chart_concepts, use_container_width=True)
        
        # التوصيات الآلية
        hardest_concepts = concept_stats[concept_stats['accuracy'] < 50]
        if not hardest_concepts.empty:
            st.error("🔴 توصية أكاديمية: يجب إعادة شرح المفاهيم الملونة بالأحمر في الحصة القادمة.")
    else:
        st.info("لا توجد بيانات مفاهيم كافية للتحليل.")

# 3. تبويب سجلات الطلاب
with tab_students:
    st.subheader("السجل الأكاديمي التفصيلي")
    
    search_term = st.text_input("البحث عن طالب:", "")
    
    # تحضير الجدول للعرض
    display_df = sum_df.copy()
    display_df = display_df.rename(columns={
        'student': 'اسم الطالب',
        'last_accuracy': 'الدرجة النهائية',
        'improvement_pct': 'نسبة التحسن',
        'avg_time_sec': 'متوسط الزمن (ث)'
    })
    
    if search_term:
        display_df = display_df[display_df['اسم الطالب'].str.contains(search_term)]
        
    st.dataframe(
        display_df[['اسم الطالب', 'الدرجة النهائية', 'نسبة التحسن', 'متوسط الزمن (ث)']].style.background_gradient(subset=['الدرجة النهائية'], cmap='RdYlGn'),
        use_container_width=True
    )

# 4. تبويب إنشاء الاختبارات
with tab_exam:
    st.subheader("أداة توليد الاختبارات المعيارية")
    c_ex1, c_ex2 = st.columns([1, 2])
    
    with c_ex1:
        st.markdown("**خصائص الاختبار**")
        target_ch = st.multiselect("المجال (الفصول):", [1, 2, 3, 4, 5], default=[1])
        q_num = st.number_input("عدد الفقرات:", min_value=1, max_value=50, value=5)
        gen_btn = st.button("إنشاء النموذج")
        
    with c_ex2:
        if gen_btn and api_key:
            with st.spinner("جاري معالجة المعايير وإنشاء الفقرات..."):
                quiz = generate_mixed_quiz(api_key, target_ch, q_num)
                if not quiz.empty:
                    st.success("تم إنشاء الاختبار بنجاح.")
                    st.dataframe(quiz[['question', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']], use_container_width=True)
                    csv = quiz.to_csv(index=False).encode('utf-8')
                    st.download_button("تصدير بصيغة CSV", csv, "generated_exam.csv", "text/csv")
                else:
                    st.error("فشل التوليد. تحقق من المدخلات.")
        elif gen_btn:
            st.error("مفتاح API مطلوب.")