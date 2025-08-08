import numpy as np

class EmbeddingEngine:
    """
    Abstraction: supports local sentence-transformers model OR OpenAI embeddings.
    Methods:
      - get_embedding(text) -> np.ndarray
      - get_embeddings(list_of_texts) -> list[np.ndarray]
      - summarize_with_openai(job_text, resume_text) -> str  # optional
    """

    def __init__(self, mode="sentence-transformers (local)", model_name="all-MiniLM-L6-v2", openai_api_key=None):
        self.mode = mode
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        self._model = None
        if self.mode.startswith("sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load sentence-transformers model '{self.model_name}': {e}")

        if self.mode.startswith("OpenAI"):
            if not openai_api_key:
                raise RuntimeError("OpenAI API key required for OpenAI mode.")
            import openai
            openai.api_key = openai_api_key

    def get_embedding(self, text):
        if self.mode.startswith("sentence-transformers"):
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec
        else:
            # OpenAI embeddings
            import openai
            resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
            return np.array(resp["data"][0]["embedding"], dtype=float)

    def get_embeddings(self, texts):
        if self.mode.startswith("sentence-transformers"):
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return [v for v in vecs]
        else:
            import openai
            # batching is recommended; here simple loop (for demo)
            embs = []
            for t in texts:
                resp = openai.Embedding.create(model="text-embedding-3-small", input=t)
                embs.append(np.array(resp["data"][0]["embedding"], dtype=float))
            return embs

    def summarize_with_openai(self, job_text, resume_text, max_tokens=150):
        """
        Use OpenAI ChatCompletion to produce a short 'why fit' summary.
        Only works if initialized in OpenAI mode.
        """
        import openai
        prompt = (
            "You are an assistant that reads a job description and a candidate resume and writes a concise "
            "summary (2-4 sentences) explaining why the candidate is a good fit for the job. "
            "Be specific: mention matching skills, years/roles if present, and relevant keywords.\n\n"
            f"Job description:\n{job_text}\n\nCandidate resume:\n{resume_text}\n\nSummary:"
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or gpt-4o / gpt-4o-mini; use available model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp["choices"][0]["message"]["content"].strip()