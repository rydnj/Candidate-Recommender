import streamlit as st
from embeddings import EmbeddingEngine
from utils import parse_resume_file, compute_similarities, rank_candidates, generate_summary_template
import os
import tempfile

st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")

st.title("Candidate Recommendation Engine — Demo")
st.markdown("Upload candidate resumes and paste a job description. The app will compute similarity and show top matches. Optional: enable OpenAI for AI summaries.")

# Sidebar: configuration
st.sidebar.header("Configuration")
engine_mode = st.sidebar.selectbox("Embedding backend", ["sentence-transformers (local)", "OpenAI (remote, needs API key)"])
model_name = st.sidebar.text_input("Local model name (sentence-transformers)", value="all-MiniLM-L6-v2")
openai_key = ""
if engine_mode.startswith("OpenAI"):
    openai_key = st.sidebar.text_input("OPENAI_API_KEY", type="password")

top_k = st.sidebar.slider("Number of results to show", 1, 10, 5)
show_summaries = st.sidebar.checkbox("Generate AI summaries (bonus)", value=True)
use_openai_for_summary = st.sidebar.checkbox("Use OpenAI for summaries (if available)", value=False)

# Inputs
st.header("1) Job Description")
job_text = st.text_area("Paste the job description here:", height=200)

st.header("2) Upload Resumes (PDF, DOCX, TXT)")
uploaded_files = st.file_uploader("Upload resume files (multiple)", accept_multiple_files=True, type=["pdf", "docx", "txt"])

st.header("3) Or paste resumes as text (optional)")
st.write("If you prefer, paste raw resume text. Each block is one resume; separate resumes with `---` on a line.")
pasted_resumes = st.text_area("Pasted resumes (use --- to separate)")

# Initialize embedding engine
engine = EmbeddingEngine(mode=engine_mode, model_name=model_name, openai_api_key=openai_key)

# Collect resumes into a list of dicts: {id, name, text}
resumes = []
# from uploaded files
if uploaded_files:
    for up in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as tmp:
                tmp.write(up.getbuffer())
                tmp.flush()
                text = parse_resume_file(tmp.name)
        finally:
            # keep file for debug if needed
            pass
        resumes.append({"id": up.name, "name": up.name, "text": text})

# from pasted
if pasted_resumes:
    blocks = [b.strip() for b in pasted_resumes.split('---') if b.strip()]
    for i, b in enumerate(blocks):
        resumes.append({"id": f"pasted_{i+1}", "name": f"Pasted resume {i+1}", "text": b})

if st.button("Run recommendation"):
    if not job_text or not resumes:
        st.error("Please provide a job description and at least one resume (upload or paste).")
        st.stop()

    with st.spinner("Computing embeddings..."):
        job_emb = engine.get_embedding(job_text)
        candidate_texts = [r["text"] for r in resumes]
        candidate_embs = engine.get_embeddings(candidate_texts)

    sims = compute_similarities(job_emb, candidate_embs)  # returns list aligned with resumes
    ranked = rank_candidates(resumes, sims)
    top = ranked[:top_k]

    st.success(f"Top {len(top)} candidates")
    for i, item in enumerate(top, start=1):
        name = item["name"]
        score = item["score"]
        st.markdown(f"### {i}. {name} — **Similarity: {score:.4f}**")
        st.write(item["text"][:1000] + ("..." if len(item["text"]) > 1000 else ""))

        if show_summaries:
            if use_openai_for_summary and engine_mode.startswith("OpenAI") and openai_key:
                try:
                    summary = engine.summarize_with_openai(job_text, item["text"])
                except Exception as e:
                    st.warning(f"OpenAI summary failed: {e}. Falling back to template.")
                    summary = generate_summary_template(job_text, item["text"])
            else:
                summary = generate_summary_template(job_text, item["text"])
            st.markdown("**Why this candidate fits (AI summary):**")
            st.write(summary)

    st.info("Done. You can tweak the model or try different resumes/job descriptions.")