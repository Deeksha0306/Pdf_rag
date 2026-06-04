# FRD — PDF RAG System

## System Architecture

```
Browser (index.html)
    │
    │  HTTP (multipart upload / JSON)
    ▼
Flask Server (app.py)
    │
    ├── PDF extraction  →  rag.extract_text()     [pypdf]
    ├── Chunking        →  rag.chunk_text()
    ├── Embedding       →  rag.get_embeddings()   [Anthropic voyage-3]
    ├── Retrieval       →  rag.retrieve()         [cosine similarity]
    └── Answer          →  rag.answer()           [Claude claude-sonnet-4]
```

---

## Modules

### `rag.py` — Core RAG Logic

#### `extract_text(pdf_path: str) -> str`
- Opens the PDF using `pypdf.PdfReader`
- Iterates over all pages, calls `page.extract_text()`
- Joins non-empty pages with double newlines
- Raises `RuntimeError` if no text is extracted (image-only PDF)

#### `chunk_text(text: str) -> list[str]`
- Splits text on whitespace into a word list
- Slides a window of `CHUNK_SIZE = 400` words with `CHUNK_OVERLAP = 60` word overlap
- Overlap prevents context from being cut at chunk boundaries
- Returns a list of string chunks

#### `get_embeddings(texts: list[str], input_type: str) -> list[list[float]]`
- Calls `POST https://api.anthropic.com/v1/embeddings`
- Model: `voyage-3`
- `input_type`: `"document"` for chunks, `"query"` for user questions
- Batches requests at 128 texts per call (API limit)
- Returns a list of float vectors (one per input text)
- Auth: `x-api-key` header from `ANTHROPIC_API_KEY` environment variable

#### `retrieve(query, chunks, chunk_embeddings) -> list[dict]`
- Embeds the query with `input_type="query"`
- Computes cosine similarity between query vector and every chunk vector
- Returns top `TOP_K = 4` results sorted by descending score
- Each result: `{ idx, score, chunk }`

#### `answer(query, retrieved) -> str`
- Formats retrieved chunks as numbered context blocks
- Calls `claude-sonnet-4` via the Anthropic Python SDK
- System prompt instructs the model to answer only from provided context
- Returns the response text

---

### `app.py` — Flask Web Server

#### `GET /`
- Renders `templates/index.html`

#### `POST /upload`
- Accepts `multipart/form-data` with a `file` field
- Validates: file present, `.pdf` extension
- Saves to a `tempfile`, runs extraction + chunking + embedding
- Stores `{ chunks, embeddings, filename, char_count }` in `DOC_STORE[session_id]`
- Deletes the temp file after processing
- Returns `{ filename, chunks, chars }` on success, `{ error }` on failure

#### `POST /ask`
- Accepts `application/json` with `{ "question": "..." }`
- Looks up session's doc store
- Calls `rag.retrieve()` then `rag.answer()`
- Returns `{ answer, sources: [{ idx, score, preview }] }`

#### `POST /clear`
- Deletes the session's entry from `DOC_STORE`
- Returns `{ ok: true }`

---

### `templates/index.html` — Frontend

- Vanilla JS, no framework or build step
- **Upload flow:** drag-and-drop or file picker → `POST /upload` → show progress + status pill → reveal chat UI
- **Chat flow:** textarea + Enter/button → `POST /ask` → append message bubbles
- **Sources:** rendered as chips below each AI answer; `title` attribute shows chunk preview on hover
- **Clear:** calls `POST /clear`, resets all UI state

---

## Configuration (constants in `rag.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `60` | Word overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `EMBED_MODEL` | `voyage-3` | Anthropic embedding model |
| `CHAT_MODEL` | `claude-sonnet-4-20250514` | Claude model for answering |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key, loaded from `.env` |

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| No file in upload request | `400 { error: "No file provided" }` |
| Non-PDF file uploaded | `400 { error: "Please upload a PDF file" }` |
| Image-only / unreadable PDF | `400 { error: "Could not extract text…" }` |
| Embedding API failure | `500 { error: "<API error message>" }` |
| Question asked before upload | `400 { error: "No document loaded…" }` |
| Missing `ANTHROPIC_API_KEY` | Server exits at startup with a clear message |

---

## Data Flow — Upload

```
User drops PDF
    → Browser POSTs file to /upload
    → pypdf extracts text (string)
    → chunk_text() splits into ~N chunks
    → get_embeddings() calls Anthropic /v1/embeddings for all chunks
    → vectors + chunks stored in DOC_STORE[session_id]
    → Browser receives { filename, chunks, chars }
    → Chat UI shown
```

## Data Flow — Question

```
User types question → Enter
    → Browser POSTs { question } to /ask
    → get_embeddings([question], "query") → query vector
    → cosine_sim(query_vector, each chunk_vector) → sorted scores
    → top 4 chunks selected
    → Claude receives: system prompt + context chunks + question
    → Claude returns answer text
    → Browser receives { answer, sources }
    → Answer bubble + source chips rendered
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | latest | Web server |
| `pypdf` | latest | PDF text extraction |
| `anthropic` | latest | Claude API (answering) |
| `httpx` | latest | HTTP client for embedding API |
| `python-dotenv` | latest | Load `.env` file |