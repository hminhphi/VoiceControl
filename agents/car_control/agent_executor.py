import json
import logging
import os
import re

import torch
from typing_extensions import override
from transformers import AutoModel, AutoTokenizer

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from graph_ql import RemoteGraphQLClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("a2a").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("car_control")

CAR_CONTROL_EMBEDDING_MODEL = os.environ.get(
    "CAR_CONTROL_EMBEDDING_MODEL",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
)
EMBEDDING_MATCH_THRESHOLD = float(os.environ.get("CAR_CONTROL_EMBEDDING_THRESHOLD", "0.2"))
ENABLE_EMBEDDING_MATCH = os.environ.get("CAR_CONTROL_ENABLE_EMBEDDING_MATCH", "").lower() in {
    "1",
    "true",
    "yes",
}

PHRASES_TO_ACTION: list[tuple[list[str], tuple[str, str]]] = [
    # English
    (["open the left door", "open left door"], ("left_door", "open")),
    (["close the left door", "close left door"], ("left_door", "close")),
    (["open the right door", "open right door"], ("right_door", "open")),
    (["close the right door", "close right door"], ("right_door", "close")),
    (["open trunk", "open the trunk"], ("trunk", "open")),
    (["close trunk", "close the trunk"], ("trunk", "close")),
    (["turn on light", "turn on the light", "turn on lights", "light on", "lights on"], ("light", "open")),
    (["turn off light", "turn off the light", "turn off lights", "light off", "lights off"], ("light", "close")),
    (["turn on ac", "turn on the ac", "turn on air conditioner", "ac on"], ("ac", "open")),
    (["turn off ac", "turn off the ac", "turn off air conditioner", "ac off"], ("ac", "close")),
    (["open window", "open windows", "open the window"], ("window", "open")),
    (["close window", "close windows", "close the window"], ("window", "close")),

    # Tiếng Việt
    (["mở cửa trái", "mở cửa bên trái", "mở cửa lái", "mở cửa tài", "mở cửa xe trái"], ("left_door", "open")),
    (["đóng cửa trái", "đóng cửa bên trái", "đóng cửa lái", "đóng cửa tài"], ("left_door", "close")),
    (["mở cửa phải", "mở cửa bên phải", "mở cửa phụ", "mở cửa xe phải"], ("right_door", "open")),
    (["đóng cửa phải", "đóng cửa bên phải", "đóng cửa phụ"], ("right_door", "close")),
    (["mở cốp", "mở cốp xe", "mở cốp sau"], ("trunk", "open")),
    (["đóng cốp", "đóng cốp xe", "đóng cốp sau"], ("trunk", "close")),
    (["bật đèn", "bật đèn xe", "mở đèn"], ("light", "open")),
    (["tắt đèn", "tắt đèn xe"], ("light", "close")),
    (["bật điều hòa", "bật máy lạnh", "mở điều hòa", "mở máy lạnh"], ("ac", "open")),
    (["tắt điều hòa", "tắt máy lạnh"], ("ac", "close")),
    (["hạ kính", "mở cửa sổ", "mở kính"], ("window", "open")),
    (["lên kính", "đóng cửa sổ", "đóng kính"], ("window", "close")),

    # 日本語 (Japanese)
    (["左のドアを開けて", "左ドアを開けて", "左のドアを開く", "左ドアオープン", "左のドアあけて"], ("left_door", "open")),
    (["左のドアを閉めて", "左ドアを閉めて", "左のドアを閉じる", "左のドアしめて"], ("left_door", "close")),
    (["右のドアを開けて", "右ドアを開けて", "右のドアを開く", "右ドアオープン", "右のドアあけて"], ("right_door", "open")),
    (["右のドアを閉めて", "右ドアを閉めて", "右のドアを閉じる", "右のドアしめて"], ("right_door", "close")),
    (["トランクを開けて", "トランクオープン", "トランクを開く", "トランクあけて"], ("trunk", "open")),
    (["トランクを閉めて", "トランクを閉じる", "トランクしめて"], ("trunk", "close")),
    (["ライトをつけて", "ライト点灯", "ヘッドライトをつけて"], ("light", "open")),
    (["ライトを消して", "ライト消灯", "ヘッドライトを消して"], ("light", "close")),
    (["エアコンをつけて", "エアコン点灯", "クーラーをつけて"], ("ac", "open")),
    (["エアコンを消して", "エアコン停止", "クーラーを消して"], ("ac", "close")),
    (["窓を開けて", "窓あけて"], ("window", "open")),
    (["窓を閉めて", "窓しめて"], ("window", "close")),
]

GRAPHQL_ACTIONS = {
    ("left_door", "open"): ("can_door_left", "open"),
    ("left_door", "close"): ("can_door_left", "close"),
    ("right_door", "open"): ("can_door_right", "open"),
    ("right_door", "close"): ("can_door_right", "close"),
    ("trunk", "open"): ("can_trunk", "open"),
    ("trunk", "close"): ("can_trunk", "close"),
    ("light", "open"): ("can_light", "open"),
    ("light", "close"): ("can_light", "close"),
    ("ac", "open"): ("can_ac", "open"),
    ("ac", "close"): ("can_ac", "close"),
    ("window", "open"): ("can_window", "open"),
    ("window", "close"): ("can_window", "close"),
}

COMPONENT_ACTION_TO_LABEL: dict[tuple[str, str], str] = {
    ("left_door", "open"): "open left door",
    ("left_door", "close"): "close left door",
    ("right_door", "open"): "open right door",
    ("right_door", "close"): "close right door",
    ("trunk", "open"): "open trunk",
    ("trunk", "close"): "close trunk",
    ("light", "open"): "turn on light",
    ("light", "close"): "turn off light",
    ("ac", "open"): "turn on ac",
    ("ac", "close"): "turn off ac",
    ("window", "open"): "open window",
    ("window", "close"): "close window",
}

EMBEDDING_ACTIONS: tuple[tuple[str, tuple[str, str]], ...] = tuple(
    (label, pair)
    for pair, label in COMPONENT_ACTION_TO_LABEL.items()
    if pair in GRAPHQL_ACTIONS
)


def _find_snapshot_dir(cache_dir: str, model_id: str) -> str | None:
    direct_name = "models--" + model_id.replace("/", "--")
    for sub in ("hub", ""):
        base = os.path.join(cache_dir, sub, direct_name) if sub else os.path.join(cache_dir, direct_name)
        snapshots = os.path.join(base, "snapshots")
        if os.path.isdir(snapshots):
            for child in os.listdir(snapshots):
                full = os.path.join(snapshots, child)
                if os.path.isdir(full) and os.listdir(full):
                    return full
    return None


class EmbeddingActionMatcher:
    def __init__(self) -> None:
        cache_dir = os.path.abspath(os.environ.get("HF_HOME", "cache"))
        model_path = _find_snapshot_dir(cache_dir, CAR_CONTROL_EMBEDDING_MODEL) or CAR_CONTROL_EMBEDDING_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir)
        self.model.eval()
        self.labels = [label for label, _ in EMBEDDING_ACTIONS]
        self.pairs = [pair for _, pair in EMBEDDING_ACTIONS]
        self.label_embeddings = self._embed(self.labels)

    def _embed(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        embeddings = summed / counts
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def best_match(self, text: str) -> tuple[tuple[str, str] | None, dict]:
        query_embedding = self._embed([text])
        similarities = torch.matmul(self.label_embeddings, query_embedding[0])
        score, idx = torch.max(similarities, dim=0)
        score_value = float(score.item())
        label = self.labels[int(idx.item())]
        pair = self.pairs[int(idx.item())]
        return pair, {
            "match_source": "embedding",
            "embedding_score": round(score_value, 4),
            "matched_label": label,
        }


def _get_user_text(context: RequestContext) -> str:
    msg = getattr(context, "message", None)
    if not msg:
        request = getattr(context, "request", None)
        if request:
            params = getattr(request, "params", None)
            logger.debug("request.params: %s", params)
            if params:
                msg = getattr(params, "message", None)
                logger.debug("params.message: %s", msg)
    if not msg:
        return ""
    parts = getattr(msg, "parts", None) or []
    logger.debug("Message parts: %s", parts)
    for p in parts:
        part_root = getattr(p, "root", p)
        kind = getattr(part_root, "kind", None) or getattr(p, "type", None)
        logger.debug("Part kind=%s, part_root=%s", kind, part_root)
        if kind == "text":
            text = getattr(part_root, "text", None) or getattr(p, "text", None) or ""
            logger.debug("Found text: %s", text)
            return text
    return ""


def _match_command_tokens(user_text: str) -> list[tuple[str, str]]:
    matched: list[tuple[str, str]] = []
    for action in ("open", "close"):
        for side, component in (("left", "left_door"), ("right", "right_door")):
            patterns = (
                rf"\b{action}\s+(?:the\s+)?{side}\s+door\b",
                rf"\b{side}\s+door\s+{action}\b",
            )
            if any(re.search(pattern, user_text, re.IGNORECASE) for pattern in patterns):
                pair = (component, action)
                if pair not in matched:
                    matched.append(pair)

        trunk_aliases = "trunk|boot|tailgate"
        patterns = (
            rf"\b{action}\s+(?:the\s+)?(?:{trunk_aliases})\b",
            rf"\b(?:{trunk_aliases})\s+{action}\b",
        )
        if any(re.search(pattern, user_text, re.IGNORECASE) for pattern in patterns):
            pair = ("trunk", action)
            if pair not in matched:
                matched.append(pair)

    return matched


def _match_actions_with_meta(
    user_text: str,
    matcher: EmbeddingActionMatcher | None,
) -> tuple[list[tuple[str, str]], dict]:
    lower = (user_text or "").strip().lower()
    if not lower:
        return [], {"match_source": "empty"}

    matched: list[tuple[str, str]] = []
    for phrases, pair in PHRASES_TO_ACTION:
        if any(p in lower for p in phrases) and pair not in matched:
            matched.append(pair)

    if matched:
        return matched, {
            "match_source": "phrase",
            "matched_label": ", ".join(
                COMPONENT_ACTION_TO_LABEL.get(pair, " ".join(pair)) for pair in matched
            ),
        }

    matched = _match_command_tokens(user_text)
    if matched:
        return matched, {
            "match_source": "lexical",
            "matched_label": ", ".join(
                COMPONENT_ACTION_TO_LABEL.get(pair, " ".join(pair)) for pair in matched
            ),
        }

    if matcher and ENABLE_EMBEDDING_MATCH:
        pair, meta = matcher.best_match(user_text)
        if pair and meta["embedding_score"] >= EMBEDDING_MATCH_THRESHOLD:
            return [pair], meta
        return [], meta

    return [], {"match_source": "none", "reason": "missing explicit action/target words"}


def _graphql_client_optional():
    if os.environ.get("GRAPHQL_HOST") and os.environ.get("GRAPHQL_API_KEY"):
        try:
            return RemoteGraphQLClient()
        except Exception as e:
            logger.warning("GraphQL client init failed: %s", e)
    return None


class CarControlAgentExecutor(AgentExecutor):
    def __init__(self, base_url: str = "") -> None:
        self._base_url = base_url or os.environ.get("CAR_CONTROL_BASE_URL", "http://localhost:8001")
        self._graphql_client = _graphql_client_optional()
        self._embedding_matcher = EmbeddingActionMatcher() if ENABLE_EMBEDDING_MATCH else None

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.debug("context: %s", context)
        text = _get_user_text(context)
        logger.info("text: %s", text)

        if not (text or "").strip():
            await event_queue.enqueue_event(
                new_agent_text_message(json.dumps({
                    "success": False,
                    "message": "No matching vehicle control action found.",
                    "meta": {"match_source": "empty"},
                }))
            )
            return

        pairs, match_meta = _match_actions_with_meta(text, self._embedding_matcher)
        if not pairs:
            await event_queue.enqueue_event(
                new_agent_text_message(json.dumps({
                    "success": False,
                    "message": "No matching vehicle control action found.",
                    "meta": match_meta,
                }))
            )
            return

        actions: list[str] = []
        for component, action in pairs:
            label = COMPONENT_ACTION_TO_LABEL.get((component, action))
            fn_name, act = GRAPHQL_ACTIONS[(component, action)]
            if self._graphql_client:
                try:
                    self._graphql_client.send_request(fn_name, {"action": act})
                except Exception as e:
                    logger.warning("GraphQL failed: %s", e)
                    continue

            # Update local car simulator if active (PC / simulation mode)
            try:
                import car_simulator as _cs_mod
                _sim = getattr(_cs_mod, "_SHARED_SIMULATOR", None)
                if _sim is not None:
                    _dispatch = {
                        "left_door":  ("set_door", action),
                        "right_door": ("set_door", action),
                        "trunk":      ("set_trunk", action),
                        "window":     ("set_window", action),
                        "light":      ("set_light", action),
                        "ac":         ("set_ac", action),
                        "mirror":     ("set_mirror", action),
                    }
                    if component in _dispatch:
                        _m, _a = _dispatch[component]
                        getattr(_sim, _m)(_a)
                from state_server import _broadcast_state
                _broadcast_state()
            except Exception as _sim_err:
                logger.debug("Simulator update skipped: %s", _sim_err)

            if label:
                actions.append(label)

        response_text = json.dumps({
            "success": bool(actions),
            "message": actions,
            "meta": match_meta,
        })
        await event_queue.enqueue_event(new_agent_text_message(response_text))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")
