# ================== Import Libraries ==================
import streamlit as st
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras
import requests
import urllib.parse
import time
from datetime import datetime
import pandas as pd
import re
import unicodedata

# Try FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

from fpdf import FPDF

DATASET_CSV_PATH = "arxiv_data_210930-054931.csv"

# ================== Page config ==================
st.set_page_config(
    page_title="Smart Research Paper Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================== Unicode Fix ==================
def fix_unicode(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    safe = text.encode("latin-1", "ignore").decode("latin-1")
    return safe

# ================== Dark CSS ==================
dark_css = """
<style>
body, .stApp {
    background-color: #0d0d0d !important;
    color: #e6e6e6 !important;
}
[data-testid="stSidebar"] {
    background-color: #111111 !important;
}
h1, h2, h3, h4, label {
    color: white !important;
    font-weight: bold !important;
}
div.stTextInput>div>input,
div.stTextArea>div>textarea {
    background-color: #d3d3d3 !important;
    color: #000 !important;
}
.paper-card {
    background-color: #141414;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid #222;
}
.paper-card:hover {
    transform: translateY(-3px);
    box-shadow: 0px 0px 20px rgba(0,255,255,0.2);
}
a { color: #66ccff !important; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# ================== Local Abstract Summarizer ==================
def simple_summarize(text, max_sentences=2, max_chars=350):
    if not text or not isinstance(text, str):
        return "No abstract available."
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    short = " ".join(s[:max_sentences])
    if len(short) > max_chars:
        short = short[:max_chars].rsplit(" ", 1)[0] + "..."
    return short

# ================== Load Models (FIXED vocab always defined) ==================
@st.cache_resource
def load_models():
    embeddings = pickle.load(open("models/embeddings.pkl", "rb"))

    sentences_raw = pickle.load(open("models/sentences.pkl", "rb"))
    sentences = list(sentences_raw)

    # Load rec model
    try:
        rec_model = torch.load("models/rec_model.pkl")
    except:
        rec_model = pickle.load(open("models/rec_model.pkl", "rb"))

    # Load text prediction model (optional)
    try:
        model = keras.models.load_model("models/model.h5")
    except:
        model = None

    # Load vectorizer + vocab safely
    vectorizer = None
    vocab = None  # <-- FIX

    try:
        with open("models/text_vectorizer_config.pkl", "rb") as f:
            vc = pickle.load(f)
        vectorizer = TextVectorization.from_config(vc)

        with open("models/text_vectorizer_weights.pkl", "rb") as f:
            vectorizer.set_weights(pickle.load(f))

        with open("models/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        vectorizer.set_vocabulary(vocab)
    except:
        vectorizer = None
        vocab = None  # <-- FIX

    return embeddings, sentences, rec_model, model, vectorizer, vocab


embeddings, sentences, rec_model, loaded_model, loaded_text_vectorizer, loaded_vocab = load_models()

# ================== Build FAISS Index ==================
def build_faiss_index(emb):
    if not FAISS_AVAILABLE or emb is None or len(emb) == 0:
        return None
    arr = np.array(emb, dtype="float32")
    faiss.normalize_L2(arr)
    dim = arr.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(arr)
    return index

faiss_index = build_faiss_index(embeddings)

# Debug panel
st.sidebar.title("📌 Debug Info")
st.sidebar.write("Embeddings:", len(embeddings))
st.sidebar.write("Titles:", len(sentences))
st.sidebar.write("FAISS Enabled:", FAISS_AVAILABLE)

# ================== Similarity Helper ==================
def similarity_score_between_texts(t1, t2):
    try:
        e1 = rec_model.encode(t1)
        e2 = rec_model.encode(t2)
        return float(util.cos_sim(e1, e2).item())
    except:
        return 0.0

# ================== Save External Papers ==================
def save_new_api_papers(new_papers):
    global embeddings, sentences, faiss_index

    try:
        df = pd.read_csv(DATASET_CSV_PATH)
    except:
        df = pd.DataFrame(columns=["terms", "titles", "abstracts"])

    if "titles" not in df.columns:
        df["titles"] = ""
    if "abstracts" not in df.columns:
        df["abstracts"] = ""

    existing = set(t.lower().strip() for t in df["titles"].tolist())
    emb_list = embeddings.tolist()
    updated = False

    for p in new_papers:
        title = p.get("title", "").strip()
        summary = p.get("summary", "")

        if not title or title.lower() in existing:
            continue

        df.loc[len(df)] = ["", title, summary]
        sentences.append(title)
        existing.add(title.lower())

        try:
            text = (title + " " + summary).strip()
            vec = rec_model.encode(text)
        except:
            vec = rec_model.encode(title)

        vec = np.array(vec, dtype="float32")
        emb_list.append(vec)

        if FAISS_AVAILABLE:
            if faiss_index is None:
                faiss_index = build_faiss_index(np.array(emb_list))
            else:
                faiss_index.add((vec / np.linalg.norm(vec)).reshape(1, -1))

        updated = True

    if updated:
        df.to_csv(DATASET_CSV_PATH, index=False)
        new_emb = np.vstack(emb_list)
        embeddings[:] = new_emb
        pickle.dump(sentences, open("models/sentences.pkl", "wb"))
        pickle.dump(new_emb, open("models/embeddings.pkl", "wb"))

# ================== Recommendation ==================
def recommendation(query, k=6):
    if not query.strip():
        return []

    try:
        q_emb = rec_model.encode(query)
    except:
        return []

    if FAISS_AVAILABLE and faiss_index is not None:
        q = q_emb.astype("float32")
        q = q / np.linalg.norm(q)
        k = min(k, len(embeddings))
        D, I = faiss_index.search(q.reshape(1, -1), k)
        out = []
        for idx, sc in zip(I[0], D[0]):
            out.append({
                "title": sentences[idx],
                "authors": "",
                "year": "",
                "url": "",
                "summary": "",
                "source": "Local",
                "score": float(sc)
            })
        return out

    sim = util.cos_sim(embeddings, q_emb)
    k = min(k, sim.shape[0])
    topk = torch.topk(sim, k=k)
    out = []
    for idx, sc in zip(topk.indices, topk.values):
        out.append({
            "title": sentences[int(idx)],
            "authors": "",
            "year": "",
            "url": "",
            "summary": "",
            "source": "Local",
            "score": float(sc.item())
        })
    return out

# ================== API Fetch ==================
def fetch_semantic_scholar(q, limit=6):
    q_enc = urllib.parse.quote(q)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={q_enc}&limit={limit}&fields=title,authors,year,url,abstract"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            out = []
            for p in r.json().get("data", []):
                out.append({
                    "title": p.get("title", ""),
                    "authors": ", ".join(a["name"] for a in p.get("authors", [])),
                    "year": p.get("year", ""),
                    "url": p.get("url", ""),
                    "summary": p.get("abstract", "")
                })
            return out, "Semantic Scholar"
    except:
        pass
    return None, None

def fetch_openalex(q, limit=6):
    q_enc = urllib.parse.quote(q)
    url = f"https://api.openalex.org/works?filter=title.search:{q_enc}&per-page={limit}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            out = []
            for p in r.json().get("results", []):
                out.append({
                    "title": p.get("display_name", ""),
                    "authors": ", ".join(a["author"]["display_name"] for a in p.get("authorships", [])),
                    "year": p.get("publication_year", ""),
                    "url": p.get("id", ""),
                    "summary": ""
                })
            return out, "OpenAlex"
    except:
        pass
    return None, None

def fetch_fallback(q, limit=6):
    ss, src = fetch_semantic_scholar(q, limit)
    if ss:
        return ss, src
    oa, src = fetch_openalex(q, limit)
    if oa:
        return oa, src
    return [], "None"

# ================== PDF Export ==================
def create_pdf_from_results(results):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 14)
    pdf.multi_cell(0, 8, fix_unicode("Smart Research Paper Recommender - Ranked Results"))
    pdf.ln(4)

    pdf.set_font("Arial", "", 11)

    for i, r in enumerate(results, 1):
        title = fix_unicode(r.get("title", ""))
        authors = fix_unicode(r.get("authors", ""))
        year = fix_unicode(str(r.get("year", "")))
        link = fix_unicode(r.get("url", ""))
        source = fix_unicode(r.get("source", ""))
        abstract = fix_unicode(r.get("summary", ""))

        pdf.multi_cell(0, 6, f"{i}. {title}")
        if authors: pdf.multi_cell(0, 5, f"Authors: {authors}")
        if year: pdf.multi_cell(0, 5, f"Year: {year}")
        if link: pdf.multi_cell(0, 5, f"Link: {link}")
        if r.get("score") is not None:
            pdf.multi_cell(0, 5, f"Similarity: {r['score']:.3f}")
        if abstract:
            pdf.multi_cell(0, 5, "Summary: " + simple_summarize(abstract))

        pdf.ln(4)

    raw = pdf.output(dest="S")
    return raw.encode("latin-1", "ignore")

# ================== Card Renderer ==================
def render_paper_card(p, idx=None, show_score=True):
    st.markdown("<div class='paper-card'>", unsafe_allow_html=True)

    title = f"{idx}. {p['title']}" if idx else p["title"]
    st.markdown(f"**{fix_unicode(title)}**")

    if p.get("url"):
        st.markdown(f"[🔗 View Paper]({p['url']})")

    meta = []
    if p.get("authors"): meta.append(p["authors"])
    if p.get("year"): meta.append(str(p["year"]))
    if meta:
        st.caption(" • ".join(meta))

    footer = f"🔹 {p.get('source','External')}"
    if show_score and p.get("score") is not None:
        footer += f" • {p['score']:.3f}"

    st.markdown(f"<small>{footer}</small>", unsafe_allow_html=True)

    if p.get("summary"):
        with st.expander("Show Abstract"):
            st.write(p["summary"])
            st.caption("Summary: " + simple_summarize(p["summary"]))

    st.markdown("</div>", unsafe_allow_html=True)

# ================== MAIN UI ==================
st.title("📚 Smart Research Paper Recommender — Dark Mode")
st.caption("Hybrid AI • Fast Search • Auto-Saving Dataset")

left, right = st.columns([1, 2])

with left:
    st.header("🔍 Search Panel")
    query = st.text_input("Enter title or keywords")
    summary_text = st.text_area("Optional: abstract (for context)", height=120)

    st.subheader("⚙ Settings")
    top_k_local = st.slider("Local Results", 1, 10, 6)
    top_k_external = st.slider("External Results", 1, 10, 6)
    show_scores = st.checkbox("Show Similarity Scores", True)

    search = st.button("🚀 Fetch Recommendations")

with right:
    if not search:
        st.info("Enter a title and click the button.")
    else:
        with st.spinner("Fetching recommendations..."):

            local = recommendation(query, top_k_local)

            ext, ext_source = fetch_fallback(query, top_k_external)
            for p in ext:
                p["source"] = ext_source
                p["score"] = similarity_score_between_texts(query, p["title"])

            save_new_api_papers(ext)

            combined = []

            for r1 in local:
                combined.append({
                    "title": r1["title"],
                    "authors": "",
                    "year": "",
                    "url": "",
                    "source": "Local",
                    "score": r1["score"],
                    "summary": ""
                })

            for p in ext:
                combined.append({
                    "title": p.get("title", ""),
                    "authors": p.get("authors", ""),
                    "year": p.get("year", ""),
                    "url": p.get("url", ""),
                    "source": p.get("source", ""),
                    "score": p.get("score", 0.0),
                    "summary": p.get("summary", "")
                })

            combined_sorted = sorted(
                combined,
                key=lambda x: x["score"] if x["score"] else 0.0,
                reverse=True
            )

        st.subheader("🔝 Ranked Results")

        if combined_sorted:
            cols = st.columns(2)
            for i, p in enumerate(combined_sorted, 1):
                with cols[(i - 1) % 2]:
                    render_paper_card(p, i, show_scores)

            pdf_bytes = create_pdf_from_results(combined_sorted)
            st.download_button(
                "📄 Download PDF",
                data=pdf_bytes,
                file_name="recommendations.pdf",
                mime="application/pdf"
            )

        else:
            st.warning("No results found. Try another query.")
