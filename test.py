# tests/quick_test.py
from embeddings import EmbeddingEngine
from utils import compute_similarities

engine = EmbeddingEngine(mode="sentence-transformers (local)")
job = "Senior Python developer with experience in ML, REST APIs, and recommendation systems."
cands = [
    "Alice: Python, Flask, recommendation systems, ML.",
    "Bob: Java developer with Spring.",
]
je = engine.get_embedding(job)
ces = engine.get_embeddings(cands)
sims = compute_similarities(je, ces)
print("Sims:", sims)
