# PDF RAG

A simple local RAG (Retrieval-Augmented Generation) system for PDFs. Upload a PDF, ask questions, get answers grounded in the document — all running on your machine.

---

## How it works

1. PDF text is extracted page-by-page using `pypdf`
2. Text is split into 400-word chunks with 60-word overlaps
3. Each chunk is embedded using Google's `gemini-embedding-001` model (real semantic vectors)
4. When you ask a question, it's also embedded and compared against all chunks via cosine similarity
5. The top 4 most relevant chunks are sent to `gemini-2.5-flash` as context
6. Gemini answers strictly from that context

---

## Project structure

```
pdf-rag/
├── rag.py              # Core logic: extraction, chunking, embedding, retrieval, answering
├── app.py              # Flask server with /upload, /ask, /clear routes
├── requirements.txt    # Python dependencies
├── .env                # Your API key (never commit this)
├── .gitignore
├── PRD.md              # Product Requirements Document
├── FRD.md              # Functional Requirements Document
└── templates/
    └── index.html      # Frontend (vanilla JS, no framework)
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Open `.env` and replace the placeholder:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com) — no credit card required.

### 3. Run

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Usage

1. Drag and drop a PDF onto the upload zone (or click to browse)
2. Wait for indexing to complete — you'll see how many chunks were created
3. Type a question and press Enter or click **Ask**
4. Hover over the source chips below each answer to preview the chunks used

---

## Configuration

All tunable constants are at the top of `rag.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `60` | Word overlap between chunks |
| `TOP_K` | `4` | Chunks retrieved per query |
| `EMBED_MODEL` | `gemini-embedding-001` | Embedding model |
| `CHAT_MODEL` | `gemini-2.5-flash` | Model used for answering |

---

## Limitations

- Text-based PDFs only — scanned/image PDFs won't work (no OCR)
- Embeddings are stored in memory and reset when the server restarts
- One document per session

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web server |
| `pypdf` | PDF text extraction |
| `httpx` | HTTP client for Gemini API calls |
| `python-dotenv` | Load `.env` file |