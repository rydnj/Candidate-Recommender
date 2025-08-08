# Candidate Recommendation Engine

## What this does

A small Streamlit app that:

- Accepts a job description (text)
- Accepts candidate resumes (PDF/DOCX/TXT or pasted text)
- Generates embeddings (sentence-transformers by default)
- Computes cosine similarity between job and each resume
- Displays top N matches with similarity scores
- Bonus: AI-generated summary describing why the person fits (via OpenAI or a heuristic fallback)

## Run locally (recommended)

1. Clone:
   git clone <repo-url>
   cd candidate-reco

2. Create venv & install:
   python -m venv venv
   source venv/bin/activate # Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Launch:
   streamlit run app.py

4. In the sidebar choose `sentence-transformers (local)` and leave model `all-MiniLM-L6-v2` (fast).

## Optional: Enable OpenAI summaries

- Choose `OpenAI` backend in the sidebar and paste your `OPENAI_API_KEY`.
- Toggle `Use OpenAI for summaries`. The app will call ChatCompletion for short 2–4 sentence summaries.

## Deployment

- Streamlit Cloud / Replit: push repo and set environment variables for OPENAI_API_KEY if used.
- Heroku / Docker: create a Procfile or Dockerfile. Streamlit Cloud is easiest for quick demos.

## Assumptions & notes

- For the interview, the local `sentence-transformers` model is sufficient and fast.
- The code normalizes embeddings when using sentence-transformers (makes cosine similarity straightforward).
- The `generate_summary_template` is a deterministic fallback (no API required).
- Example improvements for production: more robust resume parsing (NLP entity extraction), caching embeddings, user auth, pagination, larger models for higher accuracy.

## Files

- `app.py` — Streamlit entry
- `embeddings.py` — embedding backend abstraction
- `utils.py` — parsers, similarity, fallback summarizer
