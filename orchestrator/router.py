import logging
import os
import re
from pathlib import Path

import numpy as np
from typing import List
import heapq

from utils import timeit, read_config

from sentence_transformers import SentenceTransformer
from registry import AgentRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator.router")


CAR_CONTROL_COMMANDS = (
    # English
    "open left door",
    "close left door",
    "open right door",
    "close right door",
    "open trunk",
    "close trunk",
    "turn on light",
    "turn off light",
    "turn on ac",
    "turn off ac",
    "open window",
    "close window",
    # Tiếng Việt
    "mở cửa trái",
    "đóng cửa trái",
    "mở cửa phải",
    "đóng cửa phải",
    "mở cốp",
    "đóng cốp",
    "bật đèn",
    "tắt đèn",
    "bật điều hòa",
    "tắt điều hòa",
    "hạ kính",
    "lên kính",
    # 日本語
    "左のドアを開けて",
    "左のドアを閉めて",
    "右のドアを開けて",
    "右のドアを閉めて",
    "トランクを開けて",
    "トランクを閉めて",
    "ライトをつけて",
    "ライトを消して",
    "エアコンをつけて",
    "エアコンを消して",
    "窓を開けて",
    "窓を閉めて",
)

EXECUTIVE_TITLE_RE = re.compile(
    r"\b(?:"
    r"C\s*\.?\s*E\s*\.?\s*O|"
    r"C\s*\.?\s*I\s*\.?\s*O|"
    r"C\s*\.?\s*F\s*\.?\s*O|"
    r"C\s*\.?\s*O|"
    r"I\s*\.?\s*O|"
    r"E\s*\.?\s*O|"
    r"F\s*\.?\s*O|President|Resident"
    r")\.?\b",
    re.IGNORECASE,
)
MITSU_QUERY_RE = re.compile(r"\bmitsu\w*", re.IGNORECASE)
CAR_MANUAL_DEMO_QUERY_RE = re.compile(
    r"\b(?:"
    r"Tony|Cherry|"
    r"Mister|B\s*\.?\s*C|O\s*\.?\s*B\s*\.?\s*C|"
    r"logo|"
    r"FPT\s+Automotive|FPT\s+Software|"
    r"AVE|AIDV"
    r")\b",
    re.IGNORECASE,
)

def _find_snapshot_dir(cache_dir: str, model_id: str) -> str | None:
    direct_name = "models--" + model_id.replace("/", "--")
    prefixed_name = "models--sentence-transformers--" + model_id.split("/")[-1]
    for name in (direct_name, prefixed_name):
        for sub in ("hub", ""):
            base = os.path.join(cache_dir, sub, name) if sub else os.path.join(cache_dir, name)
            snapshots = os.path.join(base, "snapshots")
            if os.path.isdir(snapshots):
                for child in os.listdir(snapshots):
                    full = os.path.join(snapshots, child)
                    if os.path.isdir(full) and os.listdir(full):
                        return full
    return None


class OrchestratorRouter:
    def __init__(self, config_path: str = "config/common_config.yaml"):
        logger.info("Initializing OrchestratorRouter with config: %s", config_path)
        self.config_path = config_path
        self.config = read_config(path=self.config_path)["router"]
        self.threshold = self.config.get("threshold", 0.15)
        self.top_k = self.config.get("top_k", 3)
        self.model_id = self.config.get("model_id", "paraphrase-multilingual-MiniLM-L12-v2")
        self.cache_dir = os.path.abspath(os.environ.get("HF_HOME", "cache"))
        self.registry = AgentRegistry(self.config_path)

        self.load_model()
        self.reload_agents()
        self._load_car_control_embeddings()

        logger.info("OrchestratorRouter initialized. Agent IDs: %s", [a["agent_id"] for a in self.index])

    def load_model(self):
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        snapshot_dir = _find_snapshot_dir(self.cache_dir, self.model_id)
        if snapshot_dir is not None:
            logger.info("Loading routing embedding model from local snapshot: %s", snapshot_dir)
            self.model = SentenceTransformer(snapshot_dir)
        else:
            logger.info("Loading routing embedding model from Hugging Face: %s (cache: %s)", self.model_id, self.cache_dir)
            self.model = SentenceTransformer(self.model_id, cache_folder=self.cache_dir)
        self.model.encode(["this is dummy data"])

    def _load_car_control_embeddings(self):
        self.car_control_commands = list(CAR_CONTROL_COMMANDS)
        self.car_control_embeddings = self.model.encode(
            self.car_control_commands,
            convert_to_numpy=True,
        )

    @timeit 
    def reload_agents(self):
        self.agents = self.registry.get_all()  # Should return a list of agent dicts with 'agent_id' and 'description'
        logger.info("Loaded %d agents", len(self.agents))

        self.corpus = []
        self.index = []
        for agent in self.agents:
            aid = agent.get("agent_id")
            if not aid:
                continue
            # Start with main description
            doc_parts = [agent.get("description", "")]
            # Add skill descriptions and examples
            skills = agent.get("skills", [])
            for skill in skills:
                skill_desc = skill.get("description", "") 
                if skill_desc:
                    doc_parts.append(skill_desc)
                examples = skill.get("examples", [])
                for ex in examples:
                    if ex:
                        doc_parts.append(ex)
            # Concatenate all parts into one document
            agent_doc = "\n".join([part for part in doc_parts if part])
            self.corpus.append(agent_doc)
            self.index.append({"agent_id": aid, "description": agent_doc})

        logger.info("Encoding agent corpus (documents: %d)...", len(self.corpus))
        self.embeddings = self.model.encode(self.corpus, show_progress_bar=True, convert_to_numpy=True)
        logger.info("Embeddings ready.")


    @timeit
    async def route(self, user_message: str) -> tuple[List[str], dict]:
        logger.info("Routing message: %s", user_message)

        if (
            EXECUTIVE_TITLE_RE.search(user_message or "")
            or MITSU_QUERY_RE.search(user_message or "")
            or CAR_MANUAL_DEMO_QUERY_RE.search(user_message or "")
        ):
            logger.info(
                "[router] executive/demo override matched; selected agent_id=car_manual"
            )
            return ["car_manual"], {"car_manual": 1.0}

        q_emb = self.model.encode([user_message], convert_to_numpy=True)[0]
        scores = np.dot(self.embeddings, q_emb) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-10)
        control_scores = np.dot(self.car_control_embeddings, q_emb) / (
            np.linalg.norm(self.car_control_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-10
        )
        control_embed_idx = int(np.argmax(control_scores)) if len(control_scores) else 0
        control_embed_score = float(control_scores[control_embed_idx]) if len(control_scores) else 0.0
        control_embed_label = self.car_control_commands[control_embed_idx] if len(control_scores) else None
        # Top-k indices in the entire corpus
        top = heapq.nlargest(self.top_k, enumerate(scores), key=lambda x: x[1])

        logger.info(
            "[router] query=%r threshold=%.4f top_k=%d",
            user_message,
            float(self.threshold),
            int(self.top_k),
        )
        for rank, (idx, score) in enumerate(top, 1):
            logger.info(
                "[router] candidate rank=%d agent_id=%s score=%.4f",
                rank,
                self.index[idx]["agent_id"],
                float(score),
            )
        logger.info(
            "[router] car_control embedding_score=%.4f embedding_match=%r",
            control_embed_score,
            control_embed_label,
        )

        # Map from agent_id to best score among its corpus entries
        agent_scores = {}
        for idx, score in top:
            aid = self.index[idx]['agent_id']
            if aid not in agent_scores or score > agent_scores[aid]:
                agent_scores[aid] = score
        car_control_available = "car_control" in {agent.get("agent_id") for agent in self.agents}
        # Only boost car_control if the phrase similarity is confident (>= 0.65).
        # Unrelated short phrases (e.g. "tell me a joke", "mấy giờ rồi") randomly
        # score ~0.35-0.55 against the command list and must not trigger car_control.
        if car_control_available and control_embed_score >= 0.65:
            agent_scores["car_control"] = max(float(agent_scores.get("car_control", 0.0)), control_embed_score)


        # Filter agents by threshold, then choose the highest-scoring agent.
        threshold_agents = {
            aid: float(score)
            for aid, score in agent_scores.items()
            if float(score) >= float(self.threshold)
        }
        selected_agents = []
        if threshold_agents:
            selected_agent, selected_score = max(
                threshold_agents.items(),
                key=lambda item: item[1],
            )
            selected_agents = [selected_agent]
            logger.info(
                "[router] selected highest scoring threshold candidate agent_id=%s score=%.4f",
                selected_agent,
                selected_score,
            )
        else:
            logger.info(
                "[router] no candidate reached threshold=%.4f; treating as general / outside query for LLM",
                float(self.threshold),
            )

        for aid in selected_agents:
            logger.info(
                "[router] selected agent_id=%s score=%.4f",
                aid,
                float(agent_scores.get(aid, 0)),
            )

        # Return both the selected agent_ids and their scores
        return selected_agents, agent_scores
    

# --- Testing ---
if __name__ == "__main__":
    import asyncio

    dummy_message = "Open the door and play some music"
    config_path = "config/common_config.yaml"

    async def test_route():
        router = OrchestratorRouter(config_path)
        agent_ids = await router.route(dummy_message)
        print("Selected agent_ids:", agent_ids)

    asyncio.run(test_route())
