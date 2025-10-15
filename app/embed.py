import os, glob, json
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader  # <-- PDF support

# ---------- loaders ----------

def iter_files(src_dir: str):
    """Yield absolute paths for .md/.txt/.pdf under src_dir (recursively)."""
    exts = (".md", ".txt", ".pdf")
    for path in glob.glob(os.path.join(src_dir, "**", "*"), recursive=True):
        if os.path.isfile(path) and path.lower().endswith(exts):
            yield path

def load_text_file(path: str, chunk_size: int = 700) -> List[Dict[str, Any]]:
    docs = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size].strip()
            if chunk:
                docs.append({"path": path, "page": None, "text": chunk})
    except Exception as e:
        print("skip text", path, e)
    return docs

def load_pdf(path: str, chunk_size: int = 900) -> List[Dict[str, Any]]:
    """Extract text page-by-page; sub-chunk long pages for robustness."""
    docs: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(path)
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                txt = (page.extract_text() or "").strip()
            except Exception:
                txt = ""
            if not txt:
                continue
            for i in range(0, len(txt), chunk_size):
                piece = txt[i : i + chunk_size].strip()
                if piece:
                    docs.append({"path": path, "page": page_idx, "text": piece})
    except Exception as e:
        print("skip pdf", path, e)
    return docs

def load_documents(src_dir: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for p in iter_files(src_dir):
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            docs.extend(load_pdf(p))
        else:
            docs.extend(load_text_file(p))
    return docs

# ---------- vector store ----------

class LocalVectorStore:
    """
    Minimal vector store:
    - Embeddings saved to {index_dir}/embeddings.npy  (float32, L2-normalized)
    - Metadata  saved to {index_dir}/meta.json       (list[ {path,page,text} ])
    """
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.meta_path = os.path.join(index_dir, "meta.json")
        self.emb_path  = os.path.join(index_dir, "embeddings.npy")
        os.makedirs(index_dir, exist_ok=True)

        # Small, fast, CPU-friendly model (384-dim)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings: np.ndarray | None = None   # shape (N, 384)
        self.meta: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.emb_path) and os.path.exists(self.meta_path):
            self.embeddings = np.load(self.emb_path)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.meta = []

    def rebuild(self, src_dir: str):
        docs = load_documents(src_dir)
        if not docs:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.meta = []
            self._persist()
            return

        texts = [d["text"] for d in docs]
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,         # L2-normalize here
            convert_to_numpy=True
        ).astype("float32")

        self.embeddings = emb
        self.meta = docs
        self._persist()

    def _persist(self):
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(self.emb_path, self.embeddings)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Cosine similarity via dot product (embeddings are already normalized)."""
        if self.embeddings is None or self.embeddings.shape[0] == 0:
            return []
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")  # (1, 384)

        sims = (q @ self.embeddings.T)[0]  # cosine because both are normalized
        top_idxs = np.argsort(sims)[::-1][:top_k]

        out: List[Dict[str, Any]] = []
        for i in top_idxs:
            m = self.meta[int(i)].copy()
            m["score"] = float(sims[int(i)])
            out.append(m)
        return out

# ---------- CLI ----------

def build_index_cli(src_dir: str, out_dir: str):
    store = LocalVectorStore(index_dir=out_dir)
    store.rebuild(src_dir=src_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data")
    ap.add_argument("--out", default="app/index/store")
    args = ap.parse_args()
    build_index_cli(args.src, args.out)
    print("Index built:", args.out)
