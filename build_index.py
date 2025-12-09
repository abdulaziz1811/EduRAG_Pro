import os
import pickle
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(__file__)
PDF_PATH = os.path.join(BASE_DIR, "math.pdf")
RAG_DIR = os.path.join(BASE_DIR, "rag_data")
os.makedirs(RAG_DIR, exist_ok=True)

def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    return text

def build_index():
    if not os.path.exists(PDF_PATH):
        print(f"❌ الملف غير موجود: {PDF_PATH}")
        return

    print("🔄 جاري قراءة ملف PDF...")
    doc = fitz.open(PDF_PATH)
    chunks = []
    
    for i, page in enumerate(doc):
        text = page.get_text()
        if len(text) > 50:
            raw_chunks = text.split('\n\n') 
            for chunk in raw_chunks:
                if len(chunk.strip()) > 30:
                    chunks.append({
                        "text": chunk.strip(),
                        "normalized": normalize_arabic(chunk),
                        "page": i + 1
                    })
    
    print(f"✅ تم استخراج {len(chunks)} فقرة.")
    print("🧠 بناء مصفوفة البحث...")
    vectorizer = TfidfVectorizer()
    corpus = [c['normalized'] for c in chunks]
    matrix = vectorizer.fit_transform(corpus)
    
    with open(os.path.join(RAG_DIR, "vectorizer.pkl"), "wb") as f: pickle.dump(vectorizer, f)
    with open(os.path.join(RAG_DIR, "tfidf_matrix.pkl"), "wb") as f: pickle.dump(matrix, f)
    with open(os.path.join(RAG_DIR, "chunks.pkl"), "wb") as f: pickle.dump(chunks, f)
        
    print("🎉 تم بناء الفهرس بنجاح!")

if __name__ == "__main__":
    build_index()