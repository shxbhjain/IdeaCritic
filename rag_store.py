
# rag_store.py
import os
import pickle
import math
from typing import List, Dict, Optional, Tuple

# Lazy import so module can be imported on systems without faiss / sentence-transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_RAG = True
except Exception:
    faiss = None
    SentenceTransformer = None
    HAS_RAG = False

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_FILE = os.path.join(DATA_DIR, "faiss_meta.pkl")
EMB_MODEL_NAME = os.environ.get("RAG_EMB_MODEL", "all-MiniLM-L6-v2")


# Internal singleton for the store
_rag_singleton = None


class RagUnavailableError(RuntimeError):
    pass


class FaissStore:
    def __init__(self, emb_model_name: str = EMB_MODEL_NAME):
        if not HAS_RAG:
            raise RagUnavailableError("FAISS or sentence-transformers not installed.")
        self.model = SentenceTransformer(emb_model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        # cosine via normalized inner product
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadatas: List[Dict] = []

    def _normalize(self, arr):
        # arr: numpy array
        import numpy as np
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def add_documents(self, docs: List[Dict]):
        """
        docs: [{"id": str, "text": str, "source": str, "created_at": optional}, ...]
        """
        if not docs:
            return
        texts = [d["text"] for d in docs]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = self._normalize(emb).astype("float32")
        self.index.add(emb)
        for d in docs:
            self.metadatas.append(d)

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb).astype("float32")
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(q_emb, k=min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0 and idx < len(self.metadatas): # Check for valid index
                m = dict(self.metadatas[idx])
                m["_score"] = float(score)
                results.append(m)
        return results

    def save(self):
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        with open(FAISS_META_FILE, "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self) -> bool:
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_META_FILE):
            self.index = faiss.read_index(FAISS_INDEX_FILE)
            with open(FAISS_META_FILE, "rb") as f:
                self.metadatas = pickle.load(f)
            # Ensure index dim matches model dim
            if self.index.d != self.dim:
                # Mismatch, create new index
                self.index = faiss.IndexFlatIP(self.dim)
                self.metadatas = []
                return False
            return True
        return False

    def is_empty(self) -> bool:
        return self.index.ntotal == 0


def get_rag_store(emb_model_name: str = EMB_MODEL_NAME):
    """
    Returns a singleton FaissStore instance or None if RAG is unavailable.
    """
    global _rag_singleton
    if not HAS_RAG:
        return None
    if _rag_singleton is None:
        try:
            _rag_singleton = FaissStore(emb_model_name)
            _rag_singleton.load()
        except Exception as e:
            # If init fails (e.g., model download), set singleton to None
            print(f"Warning: Failed to initialize RAG store: {e}", flush=True)
            _rag_singleton = None
    return _rag_singleton


def index_documents(docs: List[Dict]) -> bool:
    """
    Convenience wrapper to add docs and save index.
    Each doc must be {"id": str, "text": str, "source": str, ...}
    Returns True if indexing happened, False if RAG is unavailable.
    """
    store = get_rag_store()
    if store is None:
        return False
    store.add_documents(docs)
    store.save()
    return True


# --- helpers to bootstrap from MongoDB collection (if you want) ---
def index_from_db(collection, title_field="idea_title", desc_field="idea_description", transcript_field="debate_transcript"):
    """
    Read all documents from a MongoDB collection and index them into FAISS.
    Each saved chunk id will be <mongo_id>__chunk__<n>
    Note: this is a bulk initializer you can call once when deploying.
    """
    store = get_rag_store()
    if store is None:
        raise RagUnavailableError("RAG not available (missing dependencies).")
    docs_to_add = []
    cursor = collection.find({})
    for doc in cursor:
        _id = str(doc.get("_id"))
        title = doc.get(title_field, "")
        desc = doc.get(desc_field, "")
        transcript = doc.get(transcript_field, "")
        text = f"{title}\n\n{desc}\n\n{transcript}"
        # simple whitespace chunker (keeps in sync with app chunking)
        words = text.split()
        chunk_size = 300
        overlap = 50
        i = 0
        idx = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            docs_to_add.append({"id": f"{_id}__chunk__{idx}", "text": chunk, "source": _id})
            idx += 1
            i += chunk_size - overlap
    if docs_to_add:
        store.add_documents(docs_to_add)
        store.save()
    return True


# --- Scoring function for "startup idea score" using vector similarity signals ---
def score_idea(title: str, desc: str, top_k: int = 8) -> Dict:
    """
    Returns a dict:
    {
      "score": float (0-100),
      "novelty": float (0-1),
      "avg_similarity": float (0-1),
      "std_similarity": float,
      "explanation": "text",
      "hits": [ {id, source, text, _score}, ... ]
    }

    Scoring heuristic:
     - novelty = 1 - max_similarity (if idea is very similar to existing items -> low novelty)
     - avg_similarity = mean(top_k scores)  (how much overlap with existing items)
     - diversity (std of similarities) adds small bonus if retrieved items are diverse
     - final score maps novelty and avg_similarity into 0-100:
         novelty component = novelty * 55
         familiarity component = (1 - avg_similarity) * 35  (we reward balanced familiarity)
         diversity bonus = min(std * 10, 10)
    """
    store = get_rag_store()
    if store is None or store.is_empty():
        return {"score": 0.0, "novelty": 0.0, "avg_similarity": 0.0, "std_similarity": 0.0,
                "explanation": "RAG unavailable or empty (faiss/sentence-transformers not installed, or no items indexed).", "hits": []}

    query = f"{title}\n\n{desc}"
    hits = store.search(query, top_k=top_k)
    sims = [h.get("_score", 0.0) for h in hits] if hits else []
    # similarity values are inner product of normalized vectors => cosine in [-1,1], but since normalized and text embeddings,
    # typically in [0,1). clamp for safety
    sims = [max(min(s, 1.0), -1.0) for s in sims]
    import statistics
    if sims:
        max_sim = max(sims)
        avg_sim = statistics.mean(sims)
        std_sim = statistics.pstdev(sims) if len(sims) > 1 else 0.0
    else:
        # No hits, max novelty
        max_sim = 0.0
        avg_sim = 0.0
        std_sim = 0.0

    novelty = max(0.0, 1.0 - max_sim)  # higher => more novel
    # compute numeric score mapping
    novelty_component = novelty * 55.0
    familiarity_component = (1.0 - avg_sim) * 35.0
    diversity_bonus = min(std_sim * 10.0, 10.0)
    raw_score = novelty_component + familiarity_component + diversity_bonus
    # clamp to 0-100
    score = max(0.0, min(100.0, raw_score))
    if not sims:
        score = 100.0 # If no similar items, score 100
        explanation_lines = ["No similar items found in RAG store. Max novelty score assigned."]
    else:
        # explanation text showing top hits
        explanation_lines = []
        explanation_lines.append(f"Novelty (1 - max similarity): {novelty:.3f}")
        explanation_lines.append(f"Average similarity (top {len(sims)}): {avg_sim:.3f}")
        explanation_lines.append(f"Diversity (std of sims): {std_sim:.3f}")
        explanation_lines.append(f"Score breakdown: novelty*55={novelty_component:.1f}, familiarity*(1-avg)*35={familiarity_component:.1f}, diversity_bonus={diversity_bonus:.1f}")
        explanation_lines.append(f"Final score: {score:.1f} / 100")

    # prepare hits summary
    hits_summary = []
    for h in hits:
        hits_summary.append({"id": h.get("id"), "source": h.get("source"), "text": h.get("text")[:400], "_score": h.get("_score")})

    return {
        "score": round(score, 2),
        "novelty": round(novelty, 4),
        "avg_similarity": round(avg_sim, 4),
        "std_similarity": round(std_sim, 4),
        "explanation": "\n".join(explanation_lines),
        "hits": hits_summary
    }