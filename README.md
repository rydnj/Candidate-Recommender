# Candidate Recommendation Engine

Streamlit Link: https://candidate-recommender-5mybzyswamg78vshytn3bv.streamlit.app/

A simple web app that recommends the best candidates for a given job description using text embeddings and cosine similarity.

---

## Features
- Accepts a **job description** via text input
- Accepts **candidate resumes** via text box or file upload (PDF/DOCX)
- Generates **sentence embeddings** using `sentence-transformers`
- Computes **cosine similarity** between job and resumes
- Displays **top 5–10 most relevant candidates** with:
  - Name / ID
  - Similarity score (percentage)
- **Future-ready:** Placeholder function for AI-generated "fit summaries" (OpenAI API can be integrated)

---

## Approach
1. **Embedding Generation** — Used `all-MiniLM-L6-v2` model from `sentence-transformers`.
2. **Similarity Calculation** — Used `cosine_similarity` from `scikit-learn` to rank candidates.
3. **Data Input** — Resumes can be entered as plain text or uploaded (PDF/DOCX parsing supported).
4. **Output** — Displays a sorted leaderboard of candidates based on score.

---

## Installation
```bash
pip install -r requirements.txt
```

## Files

- `app.py` — Streamlit entry
- `embeddings.py` — embedding backend abstraction
- `utils.py` — parsers, similarity, fallback summarizer


