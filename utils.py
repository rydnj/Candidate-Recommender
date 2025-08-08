import numpy as np
import pdfplumber
import docx
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def parse_resume_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(path)
    elif ext == ".docx":
        return _parse_docx(path)
    elif ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def _parse_pdf(path):
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return "\n".join(texts)

def _parse_docx(path):
    doc = docx.Document(path)
    texts = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(texts)

def compute_similarities(job_emb, candidate_embs) -> List[float]:
    """
    job_emb: 1-D array
    candidate_embs: list of 1-D arrays (or 2D array)
    returns list of cosine similarities
    """
    job = np.array(job_emb).reshape(1, -1)
    cand = np.array(candidate_embs)
    if cand.ndim == 1:
        cand = cand.reshape(1, -1)
    sims = cosine_similarity(job, cand)[0]
    return sims.tolist()

def rank_candidates(resumes, sims):
    """
    resumes: list of dicts with keys 'id', 'name', 'text'
    sims: list of float, same length
    returns list of dicts sorted desc by similarity, with 'score' key added
    """
    combined = []
    for r, s in zip(resumes, sims):
        combined.append({"id": r.get("id"), "name": r.get("name"), "text": r.get("text"), "score": float(s)})
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined

def generate_summary_template(job_text: str, resume_text: str) -> str:
    """
    Simple heuristic-based summary: find overlap of keywords and return a templated explanation.
    This is a fallback when OpenAI is not used.
    """
    import re
    # extract words
    def tokenize(text):
        tokens = re.findall(r"\b[a-zA-Z0-9\-\+#]+\b", text.lower())
        return set(tokens)
    job_tokens = tokenize(job_text)
    resume_tokens = tokenize(resume_text)
    common = job_tokens.intersection(resume_tokens)
    # pick some keywords to show
    keywords = sorted(list(common))[:10]
    kstr = ", ".join(keywords) if keywords else "relevant skills/experience"
    # micro-extract first line or useful candidate headline
    headline = resume_text.strip().split("\n")[0][:200]
    summary = (f"{headline} — This candidate matches the job on keywords: {kstr}. "
               "They demonstrate relevant experience and skills aligning with the role's requirements.")
    return summary