# Mini RAG / Chat Agent — AI‑First Workflow

A minimal, production‑ready Retrieval‑Augmented Generation (RAG) agent that:
- indexes a few local text/markdown files,
- serves a tiny chat UI + API to ask questions with context,
- documents the AI‑assisted workflow,
- includes a smoke test and Docker + Fly.io deployment files.

> Works fully **offline for embeddings & retrieval** using `sentence-transformers/all-MiniLM-L6-v2`.  
> For generation, it can use any **OpenAI‑compatible** endpoint (OpenAI, Groq API w/ proxy, local llama.cpp server, etc.).  
> If no LLM key is set, it returns a **grounded context answer fallback** (useful for demos).

---

## ✨ Features
- **FastAPI** app with HTMX/Tailwind mini UI (`/`), JSON endpoint (`/ask`), and ingestion endpoint (`/ingest`).
- **FAISS** vector index persisted to `app/index/`.
- **Local embeddings** via `sentence-transformers` (no API needed).
- **Pluggable LLM provider** (OpenAI‑compatible) with `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`.
- **Deterministic system prompt** to enforce *cite-from-context* answers.
- **Smoke test** with `pytest` to validate happy‑path.
- **Dockerfile** and **Fly.io** manifest.
- **How I used AI** log stub in `AI_NOTES.md`.

---

## 🧱 Project Structure
```
mini-rag-agent/
├─ app/
│  ├─ main.py                # FastAPI app (UI + API)
│  ├─ rag.py                 # RAG pipeline (retrieve, route, generate)
│  ├─ embed.py               # Embedding & FAISS indexing helpers
│  ├─ ai_providers.py        # OpenAI-compatible LLM wrapper + fallback
│  ├─ index/
│  │  ├─ build_index.py      # CLI to (re)build index from /data
│  │  └─ store/              # Persisted FAISS + meta
│  ├─ templates/
│  │  └─ index.html          # HTMX chat UI
│  └─ static/
│     └─ styles.css          # Minimal extras on top of Tailwind CDN
├─ data/                     # Your notes (.md/.txt) go here
│  └─ sample/                # Example content
├─ tests/
│  └─ test_smoke.py
├─ scripts/
│  └─ dev.sh                 # one-liner dev run
├─ .env.example
├─ requirements.txt
├─ Dockerfile
├─ fly.toml
├─ AI_NOTES.md               # Log how AI assisted you
└─ README.md
```

---

## 🚀 Quick Start (Local)

**1) Python env**
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

**2) (Optional) Configure LLM**
Create `.env` (or set shell env vars):
```
OPENAI_API_KEY=sk-...
# Optional if using a local/OpenAI-compatible endpoint:
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_MODEL=gpt-4o-mini
```

**3) Build the index**
```bash
python app/index/build_index.py --src data --out app/index/store
```

**4) Run the app**
```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 and ask questions!

---

## 🧪 Test
```bash
pytest -q
```

---

## 🐳 Docker
```bash
docker build -t mini-rag-agent:latest .
docker run -it --rm -p 8000:8000   -e OPENAI_API_KEY=$OPENAI_API_KEY   mini-rag-agent:latest
```
Open http://localhost:8000

---

## 🕊️ Fly.io (minimal)
> Ensure you have `flyctl` installed and are logged in.

```bash
fly launch --now --copy-config --name mini-rag-agent-$(openssl rand -hex 3)
# or: fly deploy
```
**Notes**
- Fly uses the provided `Dockerfile` & `fly.toml`.
- For private data, mount a volume or bake into the image at build time.
- Set secrets:
```bash
fly secrets set OPENAI_API_KEY=...
# (Optional) OPENAI_BASE_URL=... OPENAI_MODEL=...
```

---

## 📓 How I used AI
Record your AI‑assisted workflow here: prompts, diffs, code suggestions, and how it sped you up. See `AI_NOTES.md` for a template.

---

## 🔒 Security & Footguns
- Never ship API keys in the repo. Use env vars or Fly secrets.
- Validate user prompts if you index sensitive data.
- For large corpora, prefer chunk‑aware reranking; see TODOs in `rag.py`.

---

## ✅ Bonus: Smoke Check Script
`tests/test_smoke.py` runs a tiny end‑to‑end check (ingest + ask).

---

## 📚 Attribution
- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS
- FastAPI, HTMX, Tailwind CDN

Enjoy!
"# Mini-RAG-Agent-chatbot" 
