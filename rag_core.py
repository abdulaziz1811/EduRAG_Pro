# rag_core.py
# قلب النظام (بدون أي API) — مسؤول عن:
# - تحميل بنك الأسئلة لكل فصل
# - اختيار أسئلة المحاولة الأولى (تغطي المفاهيم)
# - اختيار أسئلة المحاولة الثانية (70% من المفاهيم الضعيفة)
# - تصحيح الإجابات وتحديد المفاهيم الضعيفة
# - تسجيل المحاولات في reports/attempts.csv
# - تسجيل ملخص لكل طالب في reports/students_summary.csv
# - تسجيل تاريخ أداء الطالب لكل مفهوم في reports/concept_history.csv
# - كشف المفاهيم التي تحتاج إعادة شرح
# - كشف الطلاب الذين يحتاجون دعم

import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
TEXTBOOK_DIR = os.path.join(BASE_DIR, "textbook")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)

TEXTBOOK_MAP = os.path.join(TEXTBOOK_DIR, "concept_pages.csv")
ATTEMPTS_CSV = os.path.join(REPORTS_DIR, "attempts.csv")
SUMMARY_CSV = os.path.join(REPORTS_DIR, "students_summary.csv")
CONCEPT_HISTORY_CSV = os.path.join(REPORTS_DIR, "concept_history.csv")


# ----------------------------- تحميل بنك الأسئلة ----------------------------- #

def _clean_correct_option(opt) -> str:
    """تنظيف correct_option من الشوائب (مثل 'a)' → 'a')."""
    if pd.isna(opt):
        return ""
    s = str(opt).strip().lower()
    for ch in s:
        if ch in ["a", "b", "c", "d"]:
            return ch
    return ""


def load_qna_for_chapter(chapter: int) -> pd.DataFrame:
    """
    تحميل questions_chX.csv + answers_chX.csv ودمجهم في DataFrame واحد.
    الأعمدة المتوقعة:
      question_id, question, option_a, option_b, option_c, option_d, concept, correct_option
    """
    q_path = os.path.join(DATA_DIR, f"questions_ch{chapter}.csv")
    a_path = os.path.join(DATA_DIR, f"answers_ch{chapter}.csv")

    if not (os.path.exists(q_path) and os.path.exists(a_path)):
        return pd.DataFrame()

    q = pd.read_csv(q_path)
    a = pd.read_csv(a_path)

    if "question_id" not in q.columns or "question_id" not in a.columns:
        return pd.DataFrame()

    df = q.merge(a, on="question_id", how="left")

    # تنظيف correct_option
    if "correct_option" not in df.columns:
        df["correct_option"] = ""
    df["correct_option"] = df["correct_option"].apply(_clean_correct_option)

    # تأكيد وجود column concept
    if "concept" not in df.columns:
        df["concept"] = "مفهوم عام"

    # ضمان نوع question_id
    df["question_id"] = df["question_id"].astype(int)

    return df


# ----------------------------- اختيار الأسئلة (RAG) ----------------------------- #

def select_first_attempt(df_qna: pd.DataFrame, num_questions: int) -> pd.DataFrame:
    """
    المحاولة الأولى:
      - نحاول نأخذ سؤال واحد على الأقل من كل concept
      - نكمل العدد عشوائياً من المتبقي
    """
    if df_qna.empty:
        return df_qna

    df = df_qna.copy()
    concepts = df["concept"].dropna().unique()
    selected_idx = []

    # سؤال واحد من كل مفهوم
    for c in concepts:
        sub = df[df["concept"] == c]
        if not sub.empty:
            idx = sub.sample(1, random_state=42).index[0]
            selected_idx.append(idx)

    remaining = num_questions - len(selected_idx)
    remaining_pool = df.drop(index=selected_idx)

    if remaining > 0 and not remaining_pool.empty:
        extra_idx = remaining_pool.sample(min(remaining, len(remaining_pool)), random_state=0).index.tolist()
        selected_idx += extra_idx

    selected = df.loc[selected_idx]

    # لو العدد أقل من المطلوب ولسه فيه أسئلة
    if len(selected) < min(num_questions, len(df)):
        extra_pool = df.drop(index=selected.index)
        extra_needed = min(num_questions - len(selected), len(extra_pool))
        if extra_needed > 0:
            selected = pd.concat([selected, extra_pool.sample(extra_needed, random_state=1)], ignore_index=False)

    # خلط الترتيب
    return selected.sample(frac=1, random_state=123).reset_index(drop=True)


def select_second_attempt(df_qna: pd.DataFrame,
                          num_questions: int,
                          weak_concepts: List[str]) -> pd.DataFrame:
    """
    المحاولة الثانية:
      - 70% أسئلة من المفاهيم الضعيفة
      - 30% عشوائي من باقي الأسئلة
    """
    df = df_qna.copy()
    if df.empty or not weak_concepts:
        return df.sample(min(num_questions, len(df))).reset_index(drop=True)

    weak_pool = df[df["concept"].isin(weak_concepts)]
    other_pool = df[~df["concept"].isin(weak_concepts)]

    num_weak = max(1, int(num_questions * 0.7))
    num_weak = min(num_weak, len(weak_pool)) if not weak_pool.empty else 0

    selected_parts = []
    if num_weak > 0:
        selected_parts.append(weak_pool.sample(num_weak, random_state=2))

    remaining = num_questions - sum(len(p) for p in selected_parts)

    if remaining > 0:
        if not other_pool.empty:
            selected_parts.append(other_pool.sample(min(remaining, len(other_pool)), random_state=3))
        elif not weak_pool.empty:
            extra_pool = weak_pool.drop(index=selected_parts[0].index)
            if not extra_pool.empty:
                selected_parts.append(extra_pool.sample(min(remaining, len(extra_pool)), random_state=4))

    if not selected_parts:
        return df.sample(min(num_questions, len(df))).reset_index(drop=True)

    selected = pd.concat(selected_parts, ignore_index=False)
    if len(selected) > num_questions:
        selected = selected.sample(num_questions, random_state=5)

    return selected.sample(frac=1, random_state=6).reset_index(drop=True)


# ----------------------------- التصحيح + إحصاءات المفاهيم ----------------------------- #

def grade_attempt(df_questions: pd.DataFrame,
                  answers: Dict[int, str]) -> Tuple[pd.DataFrame, Dict]:
    """
    يرجع:
      - res_df: صف لكل سؤال (question_id, concept, chosen_option, correct_option, is_correct)
      - summary: dict فيه total, correct, accuracy, weak_concepts, concept_stats
    """
    rows = []
    for _, row in df_questions.iterrows():
        qid = int(row["question_id"])
        chosen = answers.get(qid, None)
        correct = row.get("correct_option", "")
        concept = row.get("concept", "مفهوم عام")

        is_correct = (str(chosen).strip() == str(correct).strip())

        rows.append({
            "question_id": qid,
            "concept": concept,
            "chosen_option": chosen,
            "correct_option": correct,
            "is_correct": bool(is_correct),
        })

    res_df = pd.DataFrame(rows)
    total = len(res_df)
    correct_count = res_df["is_correct"].sum()
    accuracy = (correct_count / total * 100.0) if total > 0 else 0.0

    # إحصاءات المفاهيم
    concept_stats = res_df.groupby("concept")["is_correct"].agg(["sum", "count"])
    concept_stats["error_rate"] = 1 - concept_stats["sum"] / concept_stats["count"]
    concept_stats = concept_stats.reset_index()

    # المفاهيم الضعيفة = أي مفهوم فيه نسبة خطأ > 0 في هذه المحاولة
    weak_concepts = concept_stats[concept_stats["error_rate"] > 0]["concept"].tolist()

    summary = {
        "total": total,
        "correct": int(correct_count),
        "accuracy": accuracy,
        "weak_concepts": weak_concepts,
        "concept_stats": concept_stats,
    }
    return res_df, summary


def get_pages_for_concepts(concepts: List[str]) -> List[str]:
    """
    ربط المفاهيم بصفحات الكتاب من textbook/concept_pages.csv
    """
    if not os.path.exists(TEXTBOOK_MAP) or not concepts:
        return []
    mp = pd.read_csv(TEXTBOOK_MAP)
    pages = []
    for c in concepts:
        row = mp[mp["concept"] == c]
        if not row.empty:
            pages.append(str(row.iloc[0]["pages"]))
    return pages


# ----------------------------- الحفظ في التقارير ----------------------------- #

def save_attempt(student: str,
                 chapter: int,
                 attempt_index: int,
                 summary: Dict,
                 time_taken_sec: float):
    """
    تخزين بيانات المحاولة في:
      - attempts.csv
      - students_summary.csv
      - concept_history.csv (لكل مفهوم على حدة)
    """
    # 1) attempts.csv
    row = {
        "student": student,
        "chapter": chapter,
        "attempt": attempt_index,
        "total": summary["total"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "weak_concepts": ";".join(summary["weak_concepts"]),
        "time_sec": time_taken_sec,
    }

    if os.path.exists(ATTEMPTS_CSV):
        attempts = pd.read_csv(ATTEMPTS_CSV)
        attempts = pd.concat([attempts, pd.DataFrame([row])], ignore_index=True)
    else:
        attempts = pd.DataFrame([row])

    attempts.to_csv(ATTEMPTS_CSV, index=False)

    # 2) concept_history.csv
    concept_stats = summary.get("concept_stats")
    if isinstance(concept_stats, pd.DataFrame) and not concept_stats.empty:
        ch_rows = []
        for _, r in concept_stats.iterrows():
            concept = r["concept"]
            correct = int(r["sum"])
            count = int(r["count"])
            acc = (correct / count * 100.0) if count > 0 else 0.0
            ch_rows.append({
                "student": student,
                "chapter": chapter,
                "attempt": attempt_index,
                "concept": concept,
                "correct": correct,
                "total": count,
                "accuracy": acc,
            })
        new_ch_df = pd.DataFrame(ch_rows)

        if os.path.exists(CONCEPT_HISTORY_CSV):
            old_ch = pd.read_csv(CONCEPT_HISTORY_CSV)
            concept_hist = pd.concat([old_ch, new_ch_df], ignore_index=True)
        else:
            concept_hist = new_ch_df

        concept_hist.to_csv(CONCEPT_HISTORY_CSV, index=False)

    # 3) تحديث ملخص الطالب
    update_student_summary(student)


def update_student_summary(student: str):
    """تحديث ملخص الطالب في students_summary.csv."""
    if not os.path.exists(ATTEMPTS_CSV):
        return

    attempts = pd.read_csv(ATTEMPTS_CSV)
    stu = attempts[attempts["student"] == student]
    if stu.empty:
        return

    stu = stu.sort_values(["chapter", "attempt"])

    best_acc = stu["accuracy"].max()
    last_acc = stu.iloc[-1]["accuracy"]
    first_acc = stu.iloc[0]["accuracy"]
    improvement = last_acc - first_acc
    avg_time = stu["time_sec"].mean()

    summary_row = {
        "student": student,
        "best_accuracy": best_acc,
        "last_accuracy": last_acc,
        "improvement_pct": improvement,
        "avg_time_sec": avg_time,
    }

    if os.path.exists(SUMMARY_CSV):
        summ = pd.read_csv(SUMMARY_CSV)
        summ = summ[summ["student"] != student]
        summ = pd.concat([summ, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summ = pd.DataFrame([summary_row])

    summ.to_csv(SUMMARY_CSV, index=False)


# ----------------------------- دوال تستخدمها لوحة المعلم ----------------------------- #

def load_all_attempts() -> pd.DataFrame:
    if not os.path.exists(ATTEMPTS_CSV):
        return pd.DataFrame()
    return pd.read_csv(ATTEMPTS_CSV)


def detect_concepts_to_reteach(threshold_ratio: float = 0.4) -> List[str]:
    """
    إذا 40% أو أكثر من الطلاب ظهر عندهم مفهوم معيّن في الضعف → نوصي بإعادة شرحه.
    نستخدم concept_history.csv إن وجد، وإلا نستخدم weak_concepts في attempts.csv.
    """
    # أفضل: استخدام concept_history
    if os.path.exists(CONCEPT_HISTORY_CSV):
        ch = pd.read_csv(CONCEPT_HISTORY_CSV)
        if ch.empty:
            return []
        total_students = ch["student"].nunique()
        if total_students == 0:
            return []
        # عدد الطلاب الذين كان عندهم accuracy أقل من 100 في مفهوم معيّن
        ch["is_weak"] = ch["accuracy"] < 100.0
        df = ch[ch["is_weak"]].drop_duplicates(subset=["student", "concept"])
        counts = df.groupby("concept")["student"].nunique().reset_index()
        counts["ratio"] = counts["student"] / total_students
        return counts[counts["ratio"] >= threshold_ratio]["concept"].tolist()

    # بديل: استخدام weak_concepts من attempts.csv
    attempts = load_all_attempts()
    if attempts.empty:
        return []

    total_students = attempts["student"].nunique()
    rows = []
    for _, r in attempts.iterrows():
        weak = str(r.get("weak_concepts", "")).split(";")
        for c in weak:
            c = c.strip()
            if c:
                rows.append({"student": r["student"], "concept": c})

    if not rows or total_students == 0:
        return []

    df = pd.DataFrame(rows).drop_duplicates()
    counts = df.groupby("concept")["student"].nunique().reset_index()
    counts["ratio"] = counts["student"] / total_students
    return counts[counts["ratio"] >= threshold_ratio]["concept"].tolist()


def detect_struggling_students() -> pd.DataFrame:
    """
    "موديل" مبسط لتحديد الطلاب الذين يحتاجون دعم:
      - آخر دقة < 60%
      - أو التحسن سلبي
      - أو متوسط الوقت عالي (مثلاً > 90 ثانية)
    """
    if not os.path.exists(SUMMARY_CSV):
        return pd.DataFrame()

    summ = pd.read_csv(SUMMARY_CSV)
    if summ.empty:
        return pd.DataFrame()

    cond = (
        (summ["last_accuracy"] < 60.0) |
        (summ["improvement_pct"] < 0.0) |
        (summ["avg_time_sec"] > 90.0)
    )
    struggling = summ[cond].copy()
    if struggling.empty:
        return pd.DataFrame()

    struggling["status"] = "يحتاج دعم"
    return struggling


def load_concept_history() -> pd.DataFrame:
    if not os.path.exists(CONCEPT_HISTORY_CSV):
        return pd.DataFrame()
    return pd.read_csv(CONCEPT_HISTORY_CSV)
