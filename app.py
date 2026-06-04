"""
app.py — Flask web server for the PDF RAG system

Run:
    pip install -r requirements.txt
    # add your key to .env
    python app.py
Then open http://localhost:5000
"""

import os
import tempfile
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

import rag

app = Flask(__name__)
app.secret_key = os.urandom(24)

# in-memory store keyed by session id
DOC_STORE: dict[str, dict] = {}


def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename or not allowed(f.filename):
        return jsonify({"error": "Please upload a PDF file"}), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = rag.extract_text(tmp_path)
        if not text.strip():
            return jsonify({"error": "Could not extract text (might be a scanned/image-only PDF)"}), 400

        chunks     = rag.chunk_text(text)
        embeddings = rag.get_embeddings(chunks, input_type="document")

        sid = session.get("sid")
        if not sid:
            sid = os.urandom(16).hex()
            session["sid"] = sid

        DOC_STORE[sid] = {
            "chunks":     chunks,
            "embeddings": embeddings,
            "filename":   secure_filename(f.filename),
            "char_count": len(text),
        }

        return jsonify({"filename": secure_filename(f.filename), "chunks": len(chunks), "chars": len(text)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/ask", methods=["POST"])
def ask():
    sid = session.get("sid")
    if not sid or sid not in DOC_STORE:
        return jsonify({"error": "No document loaded. Please upload a PDF first."}), 400

    data  = request.get_json()
    query = (data or {}).get("question", "").strip()
    if not query:
        return jsonify({"error": "Empty question"}), 400

    try:
        store   = DOC_STORE[sid]
        results = rag.retrieve(query, store["chunks"], store["embeddings"])
        resp    = rag.answer(query, results)
        sources = [{"idx": r["idx"], "score": r["score"], "preview": r["chunk"][:200] + "…"} for r in results]
        return jsonify({"answer": resp, "sources": sources})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    sid = session.get("sid")
    if sid and sid in DOC_STORE:
        del DOC_STORE[sid]
    return jsonify({"ok": True})


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠  GEMINI_API_KEY not set. Add it to your .env file.")
        exit(1)
    print("✓  Gemini API key found")
    print("→  Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)