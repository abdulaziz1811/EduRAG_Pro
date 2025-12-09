# generate_questions.py
"""
Script to generate MCQ questions from math.pdf using OpenAI API.

Usage:
  1) Set OPENAI_API_KEY in your environment.
  2) Adjust CHAPTER_PAGE_RANGES to match your book.
  3) Run:  python generate_questions.py
  4) It will create questions_chX.csv and answers_chX.csv in data/
"""

import os
import json
from typing import List, Dict

import pandas as pd
from PyPDF2 import PdfReader
from openai import OpenAI

BASE_DIR = os.path.dirname(__file__)
PDF_PATH = os.path.join(BASE_DIR, "math.pdf")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------
# 1) حدّد نطاق الصفحات لكل فصل (عدّلها حسب كتابك)
#    الأرقام هنا مثال فقط، غيّرها بعد ما تتأكد من الفهرس
# ---------------------------------------------------
CHAPTER_PAGE_RANGES = {
    1: (12, 59),   # الفصل 1 من الصفحة 1 إلى 15 (شاملة)
    2: (62, 101),  # الفصل 2
    3: (104, 147),  # الفصل 3
    4: (150, 175),  # الفصل 4
    5: (177, 223), # الفصل 5
}

# كم سؤال تبغى لكل فصل
QUESTIONS_PER_CHAPTER = 15

# موديل OpenAI (غيره لو تبي)
OPENAI_MODEL = "gpt-4o-mini"


def extract_text_from_pages(pdf_path: str, start_page: int, end_page: int) -> str:
    """
    استخراج النص من نطاق صفحات (1-based inclusive).
    """
    reader = PdfReader(pdf_path)
    texts = []
    # PyPDF2 يستخدم 0-based index
    for i in range(start_page - 1, min(end_page, len(reader.pages))):
        page = reader.pages[i]
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def build_prompt(text: str, chapter: int, pages_range: tuple, num_questions: int) -> str:
    """
    نبني برومبت يطلب من الموديل توليد أسئلة من النص المعطى.
    يخرج JSON Structured.
    """
    start_p, end_p = pages_range
    return f"""
أنت معلم رياضيات محترف.

أمامك جزء من كتاب الرياضيات للفصل رقم {chapter}، من الصفحات {start_p} إلى {end_p}.
المحتوى بين العلامتين أدناه:

---------------- TEXT START ----------------
{text}
---------------- TEXT END ----------------

أريد منك أن تولّد {num_questions} سؤال اختيار من متعدد (MCQ) باللغة العربية،
تغطي أكبر قدر ممكن من مفاهيم هذا الجزء من الكتاب.

كل سؤال يجب أن يكون على الشكل التالي (بنية JSON):

[
  {{
    "question_id": 1,
    "question": "نص السؤال هنا",
    "option_a": "الاختيار أ",
    "option_b": "الاختيار ب",
    "option_c": "الاختيار ج",
    "option_d": "الاختيار د",
    "correct_option": "a" أو "b" أو "c" أو "d",
    "concept": "اسم قصير للمفهوم (مثلاً: الكسور، الجذور، الضرب)"
  }},
  ...
]

- تأكد أن كل الأسئلة من محتوى النص فقط.
- لا تضف أي شرح خارج JSON.
- أعد فقط JSON صالح (بدون تعليقات أو نص إضافي).
    """.strip()


def call_openai_for_questions(prompt: str) -> List[Dict]:
    """
    استدعاء OpenAI API وإرجاع قائمة أسئلة (قواميس).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("يرجى ضبط متغير البيئة OPENAI_API_KEY قبل التشغيل.")

    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI that outputs ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    content = completion.choices[0].message.content.strip()

    # نتوقع JSON كامل
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            # لو رجع dict بدل list
            data = [data]
        return data
    except json.JSONDecodeError:
        # في حال خرب JSON، نطبع للمراجعة
        print("Failed to parse JSON from model output:")
        print(content)
        raise


def save_questions_and_answers(chapter: int, questions: List[Dict]):
    """
    حفظ ناتج الموديل إلى:
    - data/questions_ch{chapter}.csv
    - data/answers_ch{chapter}.csv
    """
    if not questions:
        return

    # ترتيب الحقول في ملف الأسئلة
    q_rows = []
    a_rows = []

    for q in questions:
        qid = int(q.get("question_id", 0))
        q_text = q.get("question", "").strip()
        option_a = q.get("option_a", "").strip()
        option_b = q.get("option_b", "").strip()
        option_c = q.get("option_c", "").strip()
        option_d = q.get("option_d", "").strip()
        correct_option = q.get("correct_option", "").strip().lower()
        concept = q.get("concept", "").strip()

        if correct_option not in ["a", "b", "c", "d"]:
            continue  # تجاهل سؤال غير مضبوط

        q_rows.append({
            "question_id": qid,
            "question": q_text,
            "option_a": option_a,
            "option_b": option_b,
            "option_c": option_c,
            "option_d": option_d,
            "concept": concept if concept else "مفهوم عام",
        })

        a_rows.append({
            "question_id": qid,
            "correct_option": correct_option,
        })

    if not q_rows:
        print(f"[Chapter {chapter}] لا توجد أسئلة صالحة للحفظ.")
        return

    questions_df = pd.DataFrame(q_rows).sort_values("question_id")
    answers_df = pd.DataFrame(a_rows).sort_values("question_id")

    q_path = os.path.join(DATA_DIR, f"questions_ch{chapter}.csv")
    a_path = os.path.join(DATA_DIR, f"answers_ch{chapter}.csv")

    questions_df.to_csv(q_path, index=False)
    answers_df.to_csv(a_path, index=False)

    print(f"[Chapter {chapter}] Saved {len(questions_df)} questions to {q_path}")
    print(f"[Chapter {chapter}] Saved answers to {a_path}")


def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"لم يتم العثور على math.pdf في: {PDF_PATH}")

    for chapter, pages_range in CHAPTER_PAGE_RANGES.items():
        start_p, end_p = pages_range
        print(f"\n=== معالجة الفصل {chapter} (صفحات {start_p}–{end_p}) ===")

        text = extract_text_from_pages(PDF_PATH, start_p, end_p)
        if not text.strip():
            print(f"[Chapter {chapter}] لا يوجد نص مستخرج من هذه الصفحات، تخطي.")
            continue

        prompt = build_prompt(text, chapter, pages_range, QUESTIONS_PER_CHAPTER)
        questions = call_openai_for_questions(prompt)
        save_questions_and_answers(chapter, questions)


if __name__ == "__main__":
    main()
