"""Hybrid RAG Q&A Store -- BM25 + Embedding + RRF Fusion.

Optimized for Jetson Orin (low-power mode):
- BM25: keyword search, <1ms
- Embedding: semantic search, ~15ms
- Hybrid: BM25 + Embedding merged via RRF, ~16ms
- MD5-based embedding cache: skip re-embedding when data is unchanged

Configuration (environment variables):
- CAR_MANUAL_BRAND            toyota | mercedes | mmc
- CAR_MANUAL_SEARCH_MODE      bm25 | embedding | hybrid  (default: hybrid)
- CAR_MANUAL_EMBEDDING_MODEL  (default: paraphrase-multilingual-MiniLM-L12-v2)
- CAR_MANUAL_TOP_K            (default: 3)
- CAR_MANUAL_SCORE_THRESHOLD  (default: 0.7)
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("car_manual.qa_store")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.environ.get(
    "CAR_MANUAL_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_SEARCH_MODE = os.environ.get("CAR_MANUAL_SEARCH_MODE", "hybrid")
DEFAULT_TOP_K = int(os.environ.get("CAR_MANUAL_TOP_K", "3"))
DEFAULT_SCORE_THRESHOLD = float(os.environ.get("CAR_MANUAL_SCORE_THRESHOLD", "0.7"))
DEFAULT_LOG_RETRIEVAL = os.environ.get("CAR_MANUAL_LOG_RETRIEVAL", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path(__file__).resolve().parent / "cache")))


# -- helpers ---------------------------------------------------------------
def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vectors along an axis."""
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom

def _parse_brands(brand: str) -> list[str]:
    tokens = [b.strip().lower() for b in brand.split(",") if b.strip()]
    if not tokens or "all" in tokens:
        return ["mmc", "toyota", "mercedes"]
    return tokens


def _load_qa_list(brand: str) -> list[dict]:
    """Load Q&A pairs from ``data/<brand>.json`` (supports comma-separated brands e.g. 'mmc,toyota')."""
    brands = _parse_brands(brand)
    combined = []
    seen = set()
    for b in brands:
        path = DATA_DIR / f"{b}.json"
        if not path.is_file():
            logger.warning("No Q&A file for brand %s at %s", b, path)
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            pairs = []
            if isinstance(data, list):
                pairs = data
            elif isinstance(data, dict) and "pairs" in data:
                pairs = data["pairs"]
            elif isinstance(data, dict) and "qa" in data:
                pairs = data["qa"]
            for item in pairs:
                q = _get_question(item)
                if q and q.lower() not in seen:
                    seen.add(q.lower())
                    combined.append(item)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
    return combined


def _get_question(pair: dict) -> str:
    return pair.get("q", "") or pair.get("question", "")


def _get_answer(pair: dict) -> str:
    return pair.get("a", "") or pair.get("answer", "")


def _md5_file(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _tokenize(text: str) -> list[str]:
    """Lightweight tokenizer: lowercase + split on whitespace/punctuation."""
    import re
    return re.findall(r"\w+", text.lower())


def _short(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# -- BM25 wrapper -----------------------------------------------------------

class BM25Index:
    """Thin wrapper around ``rank_bm25.BM25Okapi``."""

    def __init__(self, documents: list[str]) -> None:
        from rank_bm25 import BM25Okapi
        self._tokenized = [_tokenize(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """Return list of ``(doc_index, score)`` sorted desc by score."""
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


# -- embedding helpers ------------------------------------------------------
class EmbeddingIndex:
    """Embedding-based semantic search with numpy cache (normalized)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._embeddings: np.ndarray | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            cache_path = Path(CACHE_DIR)
            # Try both plain and sentence-transformers-prefixed cache names
            candidate_cache_names = [
                "models--" + self._model_name.replace("/", "--"),
                "models--sentence-transformers--" + self._model_name.split("/")[-1],
            ]
            snapshot_dir: Path | None = None
            for name in candidate_cache_names:
                for sub in ("hub", ""):
                    base = (cache_path / sub / name) if sub else (cache_path / name)
                    snapshots = base / "snapshots"
                    if snapshots.is_dir():
                        for child in snapshots.iterdir():
                            if child.is_dir() and any(child.iterdir()):
                                snapshot_dir = child
                                break
                    if snapshot_dir is not None:
                        break
                if snapshot_dir is not None:
                    break

            t0 = time.perf_counter()
            if snapshot_dir is not None:
                # Fully offline load from local snapshot directory
                logger.info("Loading embedding model from local snapshot: %s", snapshot_dir)
                self._model = SentenceTransformer(str(snapshot_dir))
            else:
                # Online load by repo id, letting sentence-transformers/huggingface_hub populate the cache
                logger.info("Loading embedding model from Hugging Face: %s (cache: %s)", self._model_name, CACHE_DIR)
                self._model = SentenceTransformer(self._model_name, cache_folder=str(CACHE_DIR))
            logger.info("Model loaded in %.2fs", time.perf_counter() - t0)
        return self._model

    def eager_load(self) -> None:
        """Force-load the model now instead of waiting for the first query."""
        self._get_model()

    def _encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode texts to numpy embeddings, optionally L2-normalized."""
        model = self._get_model()
        try:
            # Newer sentence-transformers supports normalize_embeddings
            emb = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=normalize,
            )
        except TypeError:
            # Backward compatible: normalize manually
            emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if normalize:
                emb = _l2_normalize(emb, axis=1)
        # Ensure float32 for speed/memory
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb

    def build(self, texts: list[str], cache_path: Path | None = None) -> None:
        """Encode all texts (normalized) and optionally save to disk cache."""
        t0 = time.perf_counter()
        self._embeddings = self._encode(texts, normalize=True)
        elapsed = time.perf_counter() - t0
        logger.info("Embedded %d texts in %.2fs", len(texts), elapsed)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), self._embeddings)
            logger.info("Saved embedding cache to %s", cache_path)

    def load_cache(self, cache_path: Path) -> bool:
        """Try to load embeddings from cache. Returns True on success."""
        if cache_path.is_file():
            self._embeddings = np.load(str(cache_path))
            # Safety: if old cache not normalized, normalize now
            # (still recommend versioning cache file; see QAStore patch below)
            if self._embeddings.dtype != np.float32:
                self._embeddings = self._embeddings.astype(np.float32)
            norms = np.linalg.norm(self._embeddings, axis=1)
            if float(np.mean(norms)) > 1.5:  # heuristic: not normalized
                self._embeddings = _l2_normalize(self._embeddings, axis=1)
                logger.info("Normalized loaded embeddings (detected non-normalized cache)")
            logger.info("Loaded embedding cache: %s (%d vectors)", cache_path, len(self._embeddings))
            return True
        return False

    def search(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """Return list of (doc_index, score) sorted desc by cosine similarity."""
        if self._embeddings is None:
            return []
        q_emb = self._encode([query], normalize=True)  # shape (1, D)
        scores = (self._embeddings @ q_emb.T).flatten()  # cosine (approx) in [-1, 1]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(i), float(scores[i])) for i in top_indices]

# -- RRF Fusion -------------------------------------------------------------

def rrf_fusion(
    *result_lists: list[tuple[int, float]],
    k: int = 60,
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: merge multiple ranked lists.

    ``RRF_score(doc) = sum(1 / (k + rank_i))`` for each list.
    """
    doc_scores: dict[int, float] = {}
    for results in result_lists:
        for rank, (doc_idx, _score) in enumerate(results):
            doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]


# -- Main QAStore -----------------------------------------------------------

class QAStore:
    """Hybrid RAG Q&A Store with BM25 + Embedding + RRF Fusion.

    Parameters
    ----------
    brand : str
        Car brand name (maps to ``data/<brand>.json``).
    model_name : str
        HuggingFace model identifier for embeddings.
    search_mode : str
        One of ``bm25``, ``embedding``, ``hybrid``.
    top_k : int
        Number of results to retrieve.
    score_threshold : float
        Minimum embedding score for direct quote (below -> LLM fallback).
    """

    def __init__(
        self,
        brand: str,
        model_name: str = DEFAULT_MODEL,
        search_mode: str = DEFAULT_SEARCH_MODE,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.brand = brand
        self.search_mode = search_mode.lower().strip()
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.log_retrieval = DEFAULT_LOG_RETRIEVAL

        self._pairs: list[dict] = []
        self._questions: list[str] = []
        self._bm25: BM25Index | None = None
        self._emb: EmbeddingIndex | None = None
        self._model_name = model_name

    # -- load & index ------------------------------------------------------

    def load(self) -> None:
        """Load Q&A pairs and build search indexes."""
        self._pairs = _load_qa_list(self.brand)
        if not self._pairs:
            logger.warning("No Q&A data loaded for brand: %s", self.brand)
            return

        self._questions = [_get_question(p) for p in self._pairs]
        logger.info("Loaded %d Q&A pairs for %s", len(self._pairs), self.brand)

        # BM25 index  (bm25 or hybrid)
        if self.search_mode in ("bm25", "hybrid"):
            t0 = time.perf_counter()
            self._bm25 = BM25Index(self._questions)
            logger.info("BM25 index built in %.3fs", time.perf_counter() - t0)

        # Embedding index  (embedding or hybrid)
        if self.search_mode in ("embedding", "hybrid"):
            self._emb = EmbeddingIndex(self._model_name)
            self._build_embedding_index()
            # Eagerly load the model so the first real query has no cold-start.
            # When embeddings come from cache, the model is not yet loaded.
            self._emb.eager_load()

        # Warm-up: run a few dummy queries to pre-fill CPU/GPU caches,
        # JIT compile numpy paths, etc. so first real query is fast.
        self._warm_up()

    def _build_embedding_index(self) -> None:
        """Build or load cached embedding index using MD5 hash validation."""
        brands = _parse_brands(self.brand)
        combined_hash = hashlib.md5()
        found_any = False
        for b in brands:
            data_file = DATA_DIR / f"{b}.json"
            if data_file.is_file():
                found_any = True
                combined_hash.update(_md5_file(data_file).encode())
        if not found_any:
            return

        current_hash = combined_hash.hexdigest()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_brand = "_".join(brands)
        cache_emb = CACHE_DIR / f"{safe_brand}.emb.npy"
        cache_hash_file = CACHE_DIR / f"{safe_brand}.hash"

        # Check if cache is valid
        cached_hash = ""
        if cache_hash_file.is_file():
            cached_hash = cache_hash_file.read_text().strip()

        if cached_hash == current_hash and self._emb.load_cache(cache_emb):
            logger.info("Using cached embeddings for %s (hash match)", safe_brand)
            # Still need to load the model for query-time encoding
            return

        # Cache miss or hash mismatch -> rebuild
        logger.info("Building fresh embeddings for %s (hash: %s -> %s)", safe_brand, cached_hash[:8] if cached_hash else "none", current_hash[:8])
        self._emb.build(self._questions, cache_path=cache_emb)
        cache_hash_file.write_text(current_hash)

    # -- search ------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search for relevant Q&A pairs.

        Returns list of dicts: ``[{q, a, score, source}, ...]``
        """
        if not self._pairs:
            return []

        k = top_k or self.top_k
        query = query.strip()

        if self.search_mode == "bm25":
            results = self._search_bm25(query, k)
        elif self.search_mode == "embedding":
            results = self._search_embedding(query, k)
        else:  # hybrid
            results = self._search_hybrid(query, k)

        return results

    def _search_bm25(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        raw = self._bm25.search(query, top_k)
        if self.log_retrieval:
            logger.info(
                "[retrieval] query=%r mode=bm25 top_k=%d threshold=%.4f",
                query,
                top_k,
                self.score_threshold,
            )
            self._log_raw_results("bm25", raw, limit=top_k)
        return [
            {
                "q": self._questions[idx],
                "a": _get_answer(self._pairs[idx]),
                "score": score,
                "bm25_score": score,
                "source": "bm25",
            }
            for idx, score in raw
        ]

    def _search_embedding(self, query: str, top_k: int) -> list[dict]:
        if self._emb is None:
            return []
        raw = self._emb.search(query, top_k)
        if self.log_retrieval:
            logger.info(
                "[retrieval] query=%r mode=embedding top_k=%d threshold=%.4f",
                query,
                top_k,
                self.score_threshold,
            )
            self._log_raw_results("embedding", raw, limit=top_k)
        return [
            {
                "q": self._questions[idx],
                "a": _get_answer(self._pairs[idx]),
                "score": score,
                "source": "embedding",
            }
            for idx, score in raw
        ]

    def classify_by_embedding(
        self,
        query: str,
        candidate_questions: list[str] | None = None,
    ) -> dict | None:
        """Classify a query by choosing the highest cosine-similarity question.

        This is intentionally pure embedding comparison, without BM25/RRF or
        thresholding. It is useful for tiny fixed-choice demo sets where the
        caller wants "which known question is closest to the Whisper output?"
        """
        if not self._pairs:
            return None

        if self._emb is None:
            self._emb = EmbeddingIndex(self._model_name)
            self._build_embedding_index()
            self._emb.eager_load()

        raw = self._emb.search(query.strip(), len(self._pairs))
        if candidate_questions:
            allowed = {q.strip().lower() for q in candidate_questions}
            raw = [
                (idx, score)
                for idx, score in raw
                if self._questions[idx].strip().lower() in allowed
            ]
        if not raw:
            return None

        candidates = [
            {
                "q": self._questions[idx],
                "a": _get_answer(self._pairs[idx]),
                "score": float(score),
            }
            for idx, score in raw
        ]
        idx, score = raw[0]
        pair = self._pairs[idx]
        return {
            "q": self._questions[idx],
            "a": _get_answer(pair),
            "score": float(score),
            "source": "embedding_classifier",
            "candidates": candidates,
        }

    def _search_hybrid(self, query: str, top_k: int) -> list[dict]:
        """Run BM25 + Embedding, merge with RRF."""
        bm25_results = self._bm25.search(query, top_k * 2) if self._bm25 else []
        emb_results = self._emb.search(query, top_k * 2) if self._emb else []

        fused = rrf_fusion(bm25_results, emb_results, top_k=top_k)

        # Get embedding scores for the fused results (for threshold check)
        emb_score_map = {idx: score for idx, score in emb_results}
        bm25_score_map = {idx: score for idx, score in bm25_results}

        if self.log_retrieval:
            logger.info(
                "[retrieval] query=%r mode=hybrid top_k=%d threshold=%.4f",
                query,
                top_k,
                self.score_threshold,
            )
            self._log_raw_results("bm25", bm25_results, limit=top_k * 2)
            self._log_raw_results("embedding", emb_results, limit=top_k * 2)
            for rank, (idx, rrf_score) in enumerate(fused, 1):
                logger.info(
                    "[retrieval] fused rank=%d idx=%d rrf=%.6f emb=%.4f bm25=%.4f q=%r",
                    rank,
                    idx,
                    float(rrf_score),
                    float(emb_score_map.get(idx, 0.0)),
                    float(bm25_score_map.get(idx, 0.0)),
                    _short(self._questions[idx]),
                )

        return [
            {
                "q": self._questions[idx],
                "a": _get_answer(self._pairs[idx]),
                "score": emb_score_map.get(idx, 0.0),
                "bm25_score": bm25_score_map.get(idx, 0.0),
                "rrf_score": rrf_score,
                "source": "hybrid",
            }
            for idx, rrf_score in fused
        ]

    def _log_raw_results(
        self,
        source: str,
        raw_results: list[tuple[int, float]],
        limit: int = 6,
    ) -> None:
        if not raw_results:
            logger.info("[retrieval] %s no_results", source)
            return
        for rank, (idx, score) in enumerate(raw_results[:limit], 1):
            logger.info(
                "[retrieval] %s rank=%d idx=%d score=%.4f q=%r",
                source,
                rank,
                idx,
                float(score),
                _short(self._questions[idx]),
            )

    # -- warm-up ---------------------------------------------------------------

    def _warm_up(self) -> None:
        """Run a few dummy queries to warm up all code paths."""
        warmup_queries = ["how to open trunk", "tire pressure"]
        t0 = time.perf_counter()
        for q in warmup_queries:
            self.search(q)
        logger.info(
            "Warm-up done: %d queries in %.2fms",
            len(warmup_queries),
            (time.perf_counter() - t0) * 1000,
        )

    # -- answer (high-level) -----------------------------------------------

    def answer(self, query: str) -> dict:
        """Get answer for a query.

        Returns dict:
            - ``answer``: best document's answer text
            - ``score``: confidence score of the best match
            - ``confident``: True if score >= threshold
            - ``documents``: 1 document if confident, up to 3 if not
            - ``timing_ms``: search and total timing
        """
        t_total0 = time.perf_counter()

        t_search0 = time.perf_counter()
        results = self.search(query)
        t_search_ms = (time.perf_counter() - t_search0) * 1000.0

        if not results:
            t_total_ms = (time.perf_counter() - t_total0) * 1000.0
            if self.log_retrieval:
                logger.info(
                    "[answer] query=%r no_results mode=%s brand=%s search_ms=%.3f total_ms=%.3f",
                    query,
                    self.search_mode,
                    self.brand,
                    t_search_ms,
                    t_total_ms,
                )
            return {
                "answer": f"No answer found in the {self.brand} manual.",
                "score": 0.0,
                "confident": False,
                "documents": [],
                "timing_ms": {
                    "search": round(t_search_ms, 3),
                    "total": round(t_total_ms, 3),
                },
            }

        best = results[0]
        score = float(best.get("score", 0.0))
        confident = score >= self.score_threshold

        # Confident: return only the best match.
        # Not confident: return up to 3 documents for the caller to decide.
        if confident:
            documents = [results[0]]
        else:
            documents = results[:3]

        t_total_ms = (time.perf_counter() - t_total0) * 1000.0
        if self.log_retrieval:
            logger.info(
                "[answer] query=%r brand=%s mode=%s best_score=%.4f threshold=%.4f confident=%s docs=%d search_ms=%.3f total_ms=%.3f answer=%r",
                query,
                self.brand,
                self.search_mode,
                score,
                self.score_threshold,
                confident,
                len(documents),
                t_search_ms,
                t_total_ms,
                _short(best.get("a", "")),
            )
            for rank, doc in enumerate(documents, 1):
                logger.info(
                    "[answer] doc rank=%d source=%s score=%.4f bm25=%.4f rrf=%.6f q=%r a=%r",
                    rank,
                    doc.get("source"),
                    float(doc.get("score", 0.0)),
                    float(doc.get("bm25_score", 0.0)),
                    float(doc.get("rrf_score", 0.0)),
                    _short(doc.get("q", "")),
                    _short(doc.get("a", "")),
                )
        return {
            "answer": best["a"],
            "score": score,
            "confident": confident,
            "documents": documents,
            "timing_ms": {
                "search": round(t_search_ms, 3),
                "total": round(t_total_ms, 3),
            },
        }
if __name__ == "__main__":
    """
    White-box QAStore diagnostics runner.

    Goals:
    - Inspect data loading, tokenizer, MD5 cache, BM25 scores, embedding stats,
      (optional) hybrid fusion behavior, and final answer decision logic.
    - Compatible with QAStore signature: QAStore(brand, model_name=...)
    - Allows overriding mode/top_k/threshold via ENV injection (and best-effort
      attribute override if QAStore exposes them).
    """

    import argparse
    import logging
    import os
    import time
    import inspect
    from pathlib import Path

    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser("qa_store white-box test")
    parser.add_argument("-b", "--brand", default=os.environ.get("CAR_MANUAL_BRAND", "toyota"))
    parser.add_argument("--model", default=os.environ.get("CAR_MANUAL_EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("-q", "--query", default="reset tire pressure warning")

    # These are NOT passed into QAStore.__init__ (because your signature doesn't accept them).
    # We set ENV first, and then best-effort override store attrs if they exist.
    parser.add_argument("-m", "--mode", choices=["bm25", "embedding", "hybrid"],
                        default=os.environ.get("CAR_MANUAL_SEARCH_MODE", DEFAULT_SEARCH_MODE))
    parser.add_argument("-k", "--top-k", type=int,
                        default=int(os.environ.get("CAR_MANUAL_TOP_K", str(DEFAULT_TOP_K))))
    parser.add_argument("-t", "--threshold", type=float,
                        default=float(os.environ.get("CAR_MANUAL_SCORE_THRESHOLD", str(DEFAULT_SCORE_THRESHOLD))))
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ---------- Print runtime info ----------
    print("\n=== QAStore White-box Diagnostics ===")
    print("qa_store.py         :", Path(__file__).resolve())
    print("QAStore.__init__    :", inspect.signature(QAStore.__init__))
    print("brand              :", args.brand)
    print("model_name         :", args.model)
    print("query              :", args.query)
    print("requested mode     :", args.mode)
    print("requested top_k    :", args.top_k)
    print("requested threshold:", args.threshold)
    print()

    # ---------- Inject ENV (so code using DEFAULT_* sees them) ----------
    os.environ["CAR_MANUAL_BRAND"] = str(args.brand)
    os.environ["CAR_MANUAL_EMBEDDING_MODEL"] = str(args.model)
    os.environ["CAR_MANUAL_SEARCH_MODE"] = str(args.mode)
    os.environ["CAR_MANUAL_TOP_K"] = str(args.top_k)
    os.environ["CAR_MANUAL_SCORE_THRESHOLD"] = str(args.threshold)

    # ---------- Check data & cache paths ----------
    try:
        data_dir = DATA_DIR
        cache_dir = CACHE_DIR
    except NameError:
        data_dir = Path(__file__).resolve().parent / "data"
        cache_dir = data_dir / "cache"

    data_file = data_dir / f"{args.brand.lower()}.json"
    emb_cache = cache_dir / f"{args.brand}.emb.npy"
    hash_file = cache_dir / f"{args.brand}.hash"

    print("=== Paths ===")
    print("DATA_DIR  :", data_dir)
    print("CACHE_DIR :", cache_dir)
    print("data_file :", data_file, "exists=", data_file.exists())
    print("emb_cache :", emb_cache, "exists=", emb_cache.exists())
    print("hash_file :", hash_file, "exists=", hash_file.exists())
    print()

    # ---------- Test helper functions ----------
    print("=== Helper function checks ===")

    # _tokenize
    try:
        toks = _tokenize(args.query)
        print("_tokenize(query) ->", toks)
    except Exception as e:
        print("ERROR _tokenize:", e)

    # _load_qa_list
    try:
        pairs = _load_qa_list(args.brand)
        print("_load_qa_list(brand) count ->", len(pairs))
        if pairs:
            print(" sample Q:", (_get_question(pairs[0]) or "")[:120])
            print(" sample A:", (_get_answer(pairs[0]) or "")[:120])
    except Exception as e:
        print("ERROR _load_qa_list:", e)

    # _md5_file + cache hash match
    try:
        if data_file.exists():
            md5 = _md5_file(data_file)
            cached = hash_file.read_text().strip() if hash_file.exists() else ""
            print("_md5_file(data_file) ->", md5)
            print("cached hash file      ->", cached)
            print("hash match?           ->", (md5 == cached))
        else:
            print("skip md5: data_file not found")
    except Exception as e:
        print("ERROR _md5_file/hash check:", e)

    print()

    # ---------- Instantiate store (ONLY brand + model_name) ----------
    print("=== QAStore init + load() ===")
    store = QAStore(args.brand, model_name=args.model)

    # Best-effort override if these attributes exist (some versions store them as attributes)
    for attr_name, val in [("search_mode", args.mode), ("top_k", args.top_k), ("score_threshold", args.threshold)]:
        if hasattr(store, attr_name):
            try:
                setattr(store, attr_name, val)
                print(f"override store.{attr_name} = {val}  (supported)")
            except Exception as e:
                print(f"could not override store.{attr_name}: {e}")
        else:
            print(f"store has no attribute '{attr_name}' (likely controlled by ENV/internal)")

    t0 = time.perf_counter()
    store.load()
    print(f"load() time: {(time.perf_counter() - t0)*1000:.2f} ms")
    print()

    # ---------- Inspect internal indexes if present ----------
    print("=== Internal state ===")
    for name in ["_pairs", "_questions", "_bm25", "_emb", "_model_name"]:
        if hasattr(store, name):
            obj = getattr(store, name)
            if isinstance(obj, list):
                print(f"{name}: list(len={len(obj)})")
            else:
                print(f"{name}: {type(obj).__name__} {'(None)' if obj is None else ''}")
        else:
            print(f"{name}: (not present in this QAStore version)")
    print()

    # ---------- BM25 detailed ----------
    print("=== BM25 detailed ===")
    bm25 = getattr(store, "_bm25", None)
    questions = getattr(store, "_questions", None)
    if bm25 is None:
        print("BM25 index not available (mode might be embedding-only, or QAStore version differs).")
    else:
        t1 = time.perf_counter()
        raw = bm25.search(args.query, top_k=max(10, args.top_k * 3))
        print(f"bm25.search() time: {(time.perf_counter() - t1)*1000:.2f} ms")
        print("Top BM25 results (idx, score):", raw[:10])
        if questions:
            for i, (idx, sc) in enumerate(raw[:5], 1):
                print(f" #{i} idx={idx} score={sc:.4f} Q={questions[idx][:120]}")
    print()

    # ---------- Embedding detailed ----------
    print("=== Embedding detailed ===")
    emb = getattr(store, "_emb", None)
    if emb is None:
        print("Embedding index not available (mode might be bm25-only, or QAStore version differs).")
    else:
        # embeddings matrix stats
        E = getattr(emb, "_embeddings", None)
        if E is None:
            print("emb._embeddings is None (not built/loaded).")
        else:
            print("embeddings shape:", E.shape, "dtype:", E.dtype)
            norms = np.linalg.norm(E, axis=1)
            print("embedding L2 norms: min/mean/max =", float(norms.min()), float(norms.mean()), float(norms.max()))
            # If mean norm is not ~1.0, dot-product is not cosine; be careful with threshold.
        # query scoring
        t2 = time.perf_counter()
        raw2 = emb.search(args.query, top_k=max(10, args.top_k * 3))
        print(f"emb.search() time: {(time.perf_counter() - t2)*1000:.2f} ms")
        print("Top Embedding results (idx, score):", raw2[:10])
        if questions:
            for i, (idx, sc) in enumerate(raw2[:5], 1):
                print(f" #{i} idx={idx} score={sc:.4f} Q={questions[idx][:120]}")
    print()

    # ---------- Store search() detailed ----------
    print("=== store.search() detailed ===")
    t3 = time.perf_counter()
    results = store.search(args.query)
    print(f"store.search() time: {(time.perf_counter() - t3)*1000:.2f} ms")
    if not results:
        print("(no results)")
    else:
        for i, r in enumerate(results, 1):
            meta = {k: r.get(k) for k in ["source", "score", "rrf_score"]}
            print(f"[{i}] meta={meta}")
            print(" Q:", (r.get("q") or "")[:160])
            print(" A:", (r.get("a") or "")[:160])
            print()
    print()

    # ---------- Store answer() detailed ----------
    print("=== store.answer() detailed ===")
    t4 = time.perf_counter()
    out = store.answer(args.query)
    print(f"store.answer() time: {(time.perf_counter() - t4)*1000:.2f} ms")
    print("answer:", out.get("answer"))
    print("score :", out.get("score"))
    print("needs_llm:", out.get("needs_llm"))
    ctx = out.get("context") or []
    print("context size:", len(ctx))
    if ctx:
        print("context[0] keys:", list(ctx[0].keys()))
    print("\n=== DONE ===\n")
