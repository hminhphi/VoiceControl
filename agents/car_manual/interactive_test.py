"""
Interactive test session for Car Manual Agent.

Loads model, embeddings, and indexes ONCE at startup, then enters
a REPL loop where you can type queries and get instant results.

Usage:
    python interactive_test.py
    python interactive_test.py --brand mercedes
    python interactive_test.py --mode bm25
"""

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from qa_store import QAStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("interactive")


# -- interactive session ----------------------------------------------------

class InteractiveSession:
    """Loads everything once, then runs queries in a loop."""

    def __init__(self, brand: str, mode: str, top_k: int, threshold: float) -> None:
        self.brand = brand
        self.mode = mode
        self.top_k = top_k
        self.threshold = threshold
        self.store: QAStore | None = None

    def warm_up(self) -> None:
        """Load model, embeddings, indexes. Called once at startup."""
        print("\n--- Warming up ---")
        t0 = time.perf_counter()
        self.store = QAStore(
            brand=self.brand,
            search_mode=self.mode,
            top_k=self.top_k,
            score_threshold=self.threshold,
        )
        self.store.load()
        elapsed = time.perf_counter() - t0
        print(f"--- Warm-up done: {elapsed:.2f}s ---")
        print(f"    brand={self.brand}  mode={self.mode}  top_k={self.top_k}  threshold={self.threshold}")
        print(f"    pairs={len(self.store._pairs)}")
        print()

    def query(self, text: str) -> dict:
        """Run a single query and return result dict with timing."""
        t0 = time.perf_counter()
        result = self.store.answer(text)
        total_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": result["answer"],
            "score": result["score"],
            "confident": result["confident"],
            "total_ms": round(total_ms, 2),
            "documents": result.get("documents", []),
        }

    def print_result(self, query_text: str, res: dict) -> None:
        """Pretty-print query result."""
        docs = res["documents"]
        confident = res["confident"]

        print(f"\n  Q: {query_text}")
        print(f"  confident={confident}  score={res['score']:.4f}  "
              f"docs={len(docs)}  time={res['total_ms']}ms")

        if confident:
            print(f"  A: {res['answer'][:300]}")
        else:
            for i, doc in enumerate(docs, 1):
                print(f"    [{i}] {doc['a'][:200]}")
        print()

    def run_loop(self) -> None:
        """Interactive REPL loop."""
        print("=" * 60)
        print("  Car Manual Agent -- Interactive Test Session")
        print("  Type a question and press Enter.")
        print("  Commands: :quit  :mode <bm25|embedding|hybrid>  :brand <name>")
        print("            :threshold <0.0-1.0>  :topk <n>")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue

            # Commands
            if user_input == ":quit" or user_input == ":q":
                print("Bye!")
                break

            if user_input.startswith(":mode "):
                new_mode = user_input[6:].strip()
                if new_mode in ("bm25", "embedding", "hybrid"):
                    self.store.search_mode = new_mode
                    print(f"  -> search_mode = {new_mode}")
                else:
                    print("  -> valid modes: bm25, embedding, hybrid")
                continue

            if user_input.startswith(":threshold "):
                try:
                    val = float(user_input[11:].strip())
                    self.store.score_threshold = val
                    self.threshold = val
                    print(f"  -> threshold = {val}")
                except ValueError:
                    print("  -> invalid float")
                continue

            if user_input.startswith(":topk "):
                try:
                    val = int(user_input[6:].strip())
                    self.store.top_k = val
                    self.top_k = val
                    print(f"  -> top_k = {val}")
                except ValueError:
                    print("  -> invalid int")
                continue

            if user_input.startswith(":brand "):
                new_brand = user_input[7:].strip()
                print(f"  -> reloading for brand={new_brand}...")
                self.brand = new_brand
                self.warm_up()
                continue

            # Query
            res = self.query(user_input)
            self.print_result(user_input, res)


# -- main -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Car Manual Interactive Session")
    parser.add_argument("-b", "--brand",
                        default=os.environ.get("CAR_MANUAL_BRAND", "toyota"))
    parser.add_argument("-m", "--mode",
                        default=os.environ.get("CAR_MANUAL_SEARCH_MODE", "hybrid"),
                        choices=["bm25", "embedding", "hybrid"])
    parser.add_argument("-k", "--top-k", type=int,
                        default=int(os.environ.get("CAR_MANUAL_TOP_K", "3")))
    parser.add_argument("-t", "--threshold", type=float,
                        default=float(os.environ.get("CAR_MANUAL_SCORE_THRESHOLD", "0.7")))
    args = parser.parse_args()

    session = InteractiveSession(
        brand=args.brand,
        mode=args.mode,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    session.warm_up()
    session.run_loop()
