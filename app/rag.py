import os
import re
from typing import List, Tuple, Dict, Any

from .embed import LocalVectorStore
from .ai_providers import LLMClient

# -------- Prompts --------
QA_PROMPT = """You are a careful, helpful assistant that answers using the provided context ONLY.
Write clearly and naturally:
- Summarize and synthesize; do NOT copy long raw lines from the PDF.
- Prefer short paragraphs and bullet points. Use steps for procedures.
- If the context only partially answers, say what is known and what is not.
- Never invent facts. If it’s not present, say: "I couldn't find that in the indexed notes."
- When you use a fact, cite inline with [source: FILENAME:pPAGE] when page is known or [source: FILENAME] if page is not present.
- End with a compact "Sources:" list (unique items).

Output format (adapt as needed):
1) Direct answer (1–3 sentences).
2) Key points or steps (bullets).
3) Sources:
"""

SMALLTALK_PROMPT = """You are a friendly guide for a local RAG app. Greet the user briefly and explain how to use the app:
- Add PDFs, Markdown (.md), or text (.txt) into the /data folder.
- Click "Rebuild Index" to index new files.
- Ask questions about documents; you will cite sources like file.pdf:p12.
Keep it short, warm, and helpful. Do not mention internal code or implementation details.
"""

GENERAL_CHAT_PROMPT = """You are a helpful, up-to-date, general-purpose assistant.
- Answer naturally with clear, concise language.
- Use short paragraphs and bullets when helpful.
- If asked for opinions, be balanced and pragmatic.
- Do not fabricate citations. If the user wants document-based answers, say you can switch to "docs mode".
"""

FOLLOW_UP_PROMPT = """You are a friendly chatbot. After the assistant's answer, ask ONE short, natural follow-up
question that helps the user go deeper or clarify their goal. Keep it specific and helpful.
Avoid repeating the previous answer. Output only the question.
"""

# -------- Settings --------
MAX_CONTEXT_CHARS = 10000
SMALLTALK_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|sup|yo|hola|namaste|hii+|good (morning|afternoon|evening)|"
    r"how (are|r) (you|u)|who are you|help|what can you do|thanks|thank you)\W*$",
    re.IGNORECASE,
)
RETRIEVAL_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.12"))
FILE_HINT_PATTERN = re.compile(r"\.(pdf|md|txt)\b|p\d+\b|\bfile:?|document\b", re.IGNORECASE)

def _choose_style_tag(q: str) -> str:
    ql = (q or "").lower().strip()
    if ql.startswith(("how ", "steps", "procedure", "implement", "configure", "setup")):
        return "Preferred style: numbered steps."
    if any(kw in ql for kw in ["list", "types", "pros", "cons", "benefits", "drawbacks", "features"]):
        return "Preferred style: bullet list."
    return "Preferred style: short paragraph followed by 3–6 bullets."

class RAGPipeline:
    """
    Modes:
      - auto: smart routing (default)
      - docs: force document-grounded RAG with citations
      - chat: general open-domain chat (no citations)
    """

    def __init__(self, index_dir: str, allow_general_chat: bool = True):
        self.index_dir = index_dir
        self.store = LocalVectorStore(index_dir=index_dir)
        self.llm = LLMClient()
        self.allow_general_chat = allow_general_chat
        self._mode = "auto"

    # ---- Mode controls ----
    def set_mode(self, mode: str):
        mode = (mode or "auto").strip().lower()
        if mode in {"auto", "docs", "chat"}:
            self._mode = mode

    def reload(self, index_dir: str):
        self.index_dir = index_dir
        self.store = LocalVectorStore(index_dir=index_dir)

    # ---- Helpers ----
    def _is_smalltalk(self, query: str) -> bool:
        return bool(SMALLTALK_PATTERNS.match(query or ""))

    def _format_context(self, docs: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        total = 0
        for d in docs:
            name = os.path.basename(d.get("path", "")) or "unknown"
            page = d.get("page")
            page_tag = f":p{page}" if page is not None else ""
            chunk = (d.get("text", "") or "").strip()
            if not chunk:
                continue
            block = f"[{name}{page_tag}]\n{chunk}"
            if total + len(block) > MAX_CONTEXT_CHARS and parts:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict[str, Any]]:
        return self.store.search(query, top_k=top_k)

    def _wants_docs(self, query: str, best_score: float) -> bool:
        q = (query or "").lower()
        explicit_docs = any(
            kw in q for kw in ["cite", "citation", "according to", "from the pdf", "from the document", "source:"]
        )
        has_file_hints = bool(FILE_HINT_PATTERN.search(q))
        strong_retrieval = best_score >= RETRIEVAL_CONFIDENCE_THRESHOLD
        return explicit_docs or has_file_hints or strong_retrieval

    def _maybe_interpret_mode_switch(self, query: str) -> Tuple[bool, str]:
        q = (query or "").strip()
        m = re.match(r"^/mode\s+(auto|docs|chat)\s*$", q, flags=re.IGNORECASE)
        if m:
            self.set_mode(m.group(1).lower())
            return True, ""
        return False, q

    def _welcome_and_ask_preference(self) -> str:
        """Startup / guidance message that explicitly asks the user's preference."""
        return (
            "Hi! 👋 I can work in two ways:\n"
            "1) **Chat with your documents** — grounded answers with citations from your indexed files.\n"
            "2) **Generalized chat** — open conversation without citations.\n\n"
            "What would you like to do? (You can also type `/mode docs`, `/mode chat`, or `/mode auto` anytime.)"
        )

    def _append_follow_up(self, user_query: str, assistant_answer: str) -> str:
        """Ask one natural follow-up to keep the conversation flowing."""
        try:
            prompt = (
                f"{FOLLOW_UP_PROMPT}\n\n"
                f"User message:\n{user_query}\n\n"
                f"Assistant answer:\n{assistant_answer}\n\n"
                f"Follow-up question:"
            )
            nxt = self.llm.complete(prompt)
            if nxt:
                q = nxt.strip()
                if 3 <= len(q) <= 180:
                    return assistant_answer + "\n\n" + q
        except Exception:
            pass
        # Safe fallback
        return assistant_answer + "\n\nWould you like to continue with this, or switch modes with `/mode docs` or `/mode chat`?"

    # ---- Main entrypoint ----
    def answer(self, query: str, top_k: int = 6) -> Tuple[str, List[Dict[str, Any]]]:
        # Mode switch command?
        handled, clean_q = self._maybe_interpret_mode_switch(query)
        if handled:
            return f"Switched to **{self._mode}** mode. {self._welcome_and_ask_preference()}", []

        # No query or pure greeting → ask preference explicitly
        if not clean_q.strip() or self._is_smalltalk(clean_q):
            # Give a short help + explicit choice prompt
            guide = self.llm.complete(f"{SMALLTALK_PROMPT}\n\nUser: {clean_q}\n\nAssistant:") or ""
            message = (guide.strip() + "\n\n" + self._welcome_and_ask_preference()).strip()
            return message, []

        # Forced general chat
        if self._mode == "chat" and self.allow_general_chat:
            prompt = f"{GENERAL_CHAT_PROMPT}\n\nUser: {clean_q}\n\nAssistant:"
            out = self.llm.complete(prompt) or "I'm here to chat! Ask me anything."
            return self._append_follow_up(clean_q, out.strip()), []

        # Retrieve docs (for docs or auto)
        docs = self.retrieve(clean_q, top_k=top_k)
        best_score = float(docs[0].get("score", 0.0)) if docs else 0.0
        context = self._format_context(docs)

        # If index empty and not forced DOCS → propose modes
        if not context.strip():
            if self._mode == "docs":
                msg = ("Your index is empty. Put PDFs/.md/.txt inside the /data folder and click “Rebuild Index”. "
                       "Then ask questions about your documents.")
                return msg, []
            # Auto/chat fallback
            if self.allow_general_chat:
                base = (
                    "I don't see any indexed documents yet. "
                    "Would you like **generalized chat** for now, or add files and choose **chat with documents**?"
                )
                prompt = f"{GENERAL_CHAT_PROMPT}\n\nUser: {clean_q}\n\nAssistant:"
                out = self.llm.complete(prompt) or "Sure, I can help in general. What would you like to discuss?"
                final = base + "\n\n" + out.strip() + "\n\n" + "Reply with `/mode chat` or `/mode docs`."
                return final, []
            else:
                msg = ("Your index is empty. Put PDFs/.md/.txt inside the /data folder and click “Rebuild Index”. "
                       "Then ask questions about your documents.")
                return msg, []

        # AUTO routing: if it doesn't obviously want docs, offer the choice and default to chat
        if self._mode == "auto" and self.allow_general_chat and not self._wants_docs(clean_q, best_score):
            choice = (
                "Do you want to **chat with your documents** (with citations) or have a **generalized chat**?\n"
                "You can also switch anytime with `/mode docs` or `/mode chat`."
            )
            prompt = f"{GENERAL_CHAT_PROMPT}\n\nUser: {clean_q}\n\nAssistant:"
            out = (self.llm.complete(prompt) or "I can help in general. What would you like to explore?").strip()
            return self._append_follow_up(clean_q, out + "\n\n" + choice), []

        # Normal RAG QA (docs or auto routed to docs)
        style_tag = _choose_style_tag(clean_q)
        prompt = (
            f"{QA_PROMPT}\n\n{style_tag}\n\n"
            f"# Question\n{clean_q}\n\n"
            f"# Context\n{context}\n\n"
            f"# Your Answer"
        )
        out = self.llm.complete(prompt)
        if out is None:
            # Extractive fallback when no LLM key present
            best = docs[0] if docs else {}
            text = (best.get("text", "") or "")
            ans = (text[:800] + ("..." if len(text) > 800 else "")) or "No context found."
            return ans, docs

        return self._append_follow_up(clean_q, out.strip()), docs
