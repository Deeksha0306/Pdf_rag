"""
rag.py — core RAG logic (Gemini API)
  - PDF text extraction      (pypdf)
  - chunking with overlap
  - embeddings               (Gemini text-embedding-004)
  - cosine similarity retrieval
  - answering                (gemini-2.0-flash)
"""

import math
import os
import httpx
from pypdf import PdfReader

# ── config ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400                  # words per chunk
CHUNK_OVERLAP = 60                   # word overlap between consecutive chunks
TOP_K         = 4                    # chunks returned per query
EMBED_MODEL      = "gemini-embedding-001"
CHAT_MODEL       = "gemini-2.5-flash"
GEMINI_BASE      = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_BASE_EMB  = "https://generativelanguage.googleapis.com/v1beta"


# ── PDF extraction ───────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages  = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


# ── chunking ─────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── embeddings ───────────────────────────────────────────────────────────
def _embed_one(text: str, task_type: str) -> list[float]:
    """
    Embed a single text via Gemini embedContent API.
    task_type: "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
    """
    api_key = os.environ["GEMINI_API_KEY"]
    url     = f"{GEMINI_BASE_EMB}/models/{EMBED_MODEL}:embedContent?key={api_key}"
    payload = {
        "model":   f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
    }
    r = httpx.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Embedding error {r.status_code}: {r.text}")
    return r.json()["embedding"]["values"]


def get_embeddings(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """
    Embed a list of texts.
    input_type: "document" for chunks, "query" for questions.
    Gemini embedContent is one text per call, so we loop.
    """
    task_type = "RETRIEVAL_DOCUMENT" if input_type == "document" else "RETRIEVAL_QUERY"
    return [_embed_one(t, task_type) for t in texts]


# ── cosine similarity ────────────────────────────────────────────────────
def cosine_sim(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


# ── retrieval ────────────────────────────────────────────────────────────
def retrieve(query: str, chunks: list[str], chunk_embeddings: list[list[float]]) -> list[dict]:
    q_emb  = get_embeddings([query], input_type="query")[0]
    scored = sorted(
        enumerate(chunk_embeddings),
        key=lambda t: cosine_sim(q_emb, t[1]),
        reverse=True,
    )
    return [
        {"idx": idx, "score": round(cosine_sim(q_emb, emb), 4), "chunk": chunks[idx]}
        for idx, emb in scored[:TOP_K]
    ]


# ── answer generation ────────────────────────────────────────────────────
def answer(query: str, retrieved: list[dict]) -> str:
    api_key = os.environ["GEMINI_API_KEY"]
    url     = f"{GEMINI_BASE}/models/{CHAT_MODEL}:generateContent?key={api_key}"

    context = "\n\n---\n\n".join(
        f"[Chunk {r['idx'] + 1}]:\n{r['chunk']}" for r in retrieved
    )
    system = (
        "You are a helpful assistant. Answer the user's question using only "
        "the document context provided. If the answer isn't in the context, "
        "say so clearly. Be concise and accurate."
    )
    prompt = f"{system}\n\nDocument context:\n\n{context}\n\n---\n\nQuestion: {query}"

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = httpx.post(url, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini error {r.status_code}: {r.text}")

    return r.json()["candidates"][0]["content"]["parts"][0]["text"]