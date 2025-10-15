import os
import traceback
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from .rag import RAGPipeline
from .index.build_index import build_index_cli

load_dotenv()

app = FastAPI(title="Mini RAG / Chat Agent", version="1.0.0")

# Static & templates
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
    name="static",
)
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

# Init RAG
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(os.path.dirname(__file__), "index", "store"))
DATA_DIR  = os.getenv("DATA_DIR",  os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
rag = RAGPipeline(index_dir=INDEX_DIR)

# -----------------------
# Helpers
# -----------------------
def _is_htmx(request: Request) -> bool:
    # HTMX sends HX-Request: true
    return request.headers.get("HX-Request", "").lower() == "true"

def _bubble(role: str, html: str) -> str:
    is_user = (role == "user")
    align = "justify-end" if is_user else "justify-start"
    bg = "bg-indigo-600 text-white" if is_user else "bg-slate-100"
    return f"""
    <div class="flex {align}">
      <div class="max-w-[80%] rounded-xl px-3 py-2 {bg} whitespace-pre-wrap leading-relaxed">
        {html}
      </div>
    </div>
    """

# -----------------------
# Routes
# -----------------------
@app.get("/health")
async def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class AskPayload(BaseModel):
    query: str
    top_k: int = 4

@app.post("/ask")
async def ask(payload: AskPayload):
    answer, sources = rag.answer(payload.query, top_k=payload.top_k)
    return JSONResponse({"answer": answer, "sources": sources})

@app.post("/ingest")
async def ingest(request: Request):
    """
    Rebuild index from DATA_DIR.
    - If called via HTMX, returns a short HTML snippet for in-page status.
    - Otherwise returns JSON.
    """
    try:
        build_index_cli(DATA_DIR, INDEX_DIR)
        rag.reload(index_dir=INDEX_DIR)

        if _is_htmx(request):
            # Nice inline status line for the page
            html = (
                f"<span class='text-green-700'>Index rebuilt from "
                f"<code class='font-mono'>{DATA_DIR}</code></span>"
            )
            return HTMLResponse(html)
        return {"ok": True, "indexed_dir": DATA_DIR}
    except Exception:
        err = traceback.format_exc()
        if _is_htmx(request):
            html = (
                "<span class='text-red-700 font-medium'>Index rebuild failed.</span>"
                f"<pre class='text-xs mt-1 whitespace-pre-wrap'>{err}</pre>"
            )
            return HTMLResponse(html, status_code=200)
        return JSONResponse({"ok": False, "error": err}, status_code=500)

# --- Legacy mini form endpoint (still available if you link to it) ---
@app.post("/_ask_htmx", response_class=HTMLResponse)
async def ask_htmx(query: str = Form(...)):
    answer, sources = rag.answer(query, top_k=4)
    html = "<div class='space-y-2'>"
    html += f"<div class='font-medium'>Answer</div><div class='p-3 rounded bg-gray-50 whitespace-pre-wrap'>{answer}</div>"
    if sources:
        html += "<div class='font-medium mt-4'>Sources</div><ul class='list-disc pl-6'>"
        for s in sources:
            html += f"<li><span class='font-mono text-sm'>{s.get('path','')}</span> — score={s.get('score',0):.3f}</li>"
        html += "</ul>"
    html += "</div>"
    return HTMLResponse(html)

# =========================
# Chat-style HTMX endpoint
# =========================
@app.post("/_chat_htmx", response_class=HTMLResponse)
async def chat_htmx(query: str = Form(...)):
    """
    Returns two chat bubbles (user + assistant) as an HTML snippet.
    If anything goes wrong, we render an error bubble instead of 500.
    """
    try:
        answer, sources = rag.answer(query, top_k=4)

        # Render sources as small chips (filename and optional page)
        chips = ""
        if sources:
            chips = "<div class='mt-2 flex flex-wrap gap-2 text-xs'>"
            for s in sources:
                page = s.get("page")
                page_part = f"(p{page})" if page else ""
                fname = os.path.basename(s.get("path", "")) or "unknown"
                chips += (
                    f"<span class='px-2 py-1 rounded bg-white border text-slate-600 font-mono'>"
                    f"{fname}{page_part}</span>"
                )
            chips += "</div>"

        return HTMLResponse(_bubble("user", query) + _bubble("assistant", answer + chips))
    except Exception:
        err = traceback.format_exc()
        return HTMLResponse(
            _bubble("user", query) + _bubble("assistant", "Error:\n\n" + err),
            status_code=200,
        )
