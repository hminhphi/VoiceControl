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
    scored: list[tuple[int, tuple[str, str]]] = []
    for action in ("open", "close"):
        for side, component in (("left", "left_door"), ("right", "right_door")):
            patterns = (
                rf"\b{action}\s+(?:the\s+)?{side}\s+door\b",
                rf"\b{side}\s+door\s+{action}\b",
            )
            best = None
            for pattern in patterns:
                m = re.search(pattern, user_text, re.IGNORECASE)
                if m and (best is None or m.start() < best):
                    best = m.start()
            if best is not None:
                scored.append((best, (component, action)))

        trunk_aliases = "trunk|boot|tailgate"
        patterns = (
            rf"\b{action}\s+(?:the\s+)?(?:{trunk_aliases})\b",
            rf"\b(?:{trunk_aliases})\s+{action}\b",
        )
        best = None
        for pattern in patterns:
            m = re.search(pattern, user_text, re.IGNORECASE)
            if m and (best is None or m.start() < best):
                best = m.start()
        if best is not None:
            scored.append((best, ("trunk", action)))

    scored.sort(key=lambda t: t[0])
    out: list[tuple[str, str]] = []
    for _, pair in scored:
        if pair not in out:
            out.append(pair)
    return out


# Elliptical / coordinated command expansions: rewrite compound forms into
# concatenated single commands so phrase/lexical matching finds EVERY action.
# e.g. "open left and right door" -> "open left door open right door"
_ELLIPTICAL_RES: list[tuple[re.Pattern, "callable"]] = [
    # open left and right door(s) / close the left and the right doors
    (
        re.compile(
            r"\b(open|close)\s+(?:the\s+)?(left|right)\s+and\s+(?:the\s+)?(left|right)\s+(door|window)s?\b",
            re.IGNORECASE,
        ),
        lambda m: (
            f"{m.group(1)} {m.group(2)} {m.group(4)} "
            f"{m.group(1)} {m.group(3)} {m.group(4)}"
        ),
    ),
    # open both doors / close all doors / open the two doors
    (
        re.compile(
            r"\b(open|close)\s+(?:the\s+)?(?:both|all|two)\s+doors?\b",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group(1)} left door {m.group(1)} right door",
    ),
    # turn on light and ac / turn off the lights and the ac
    (
        re.compile(
            r"\b(turn on|turn off)\s+(?:the\s+)?(lights?|ac)\s+and\s+(?:the\s+)?(lights?|ac)\b",
            re.IGNORECASE,
        ),
        lambda m: (
            f"{m.group(1)} {m.group(2).rstrip('s') if m.group(2).lower() != 'ac' else 'ac'} "
            f"{m.group(1)} {m.group(3).rstrip('s') if m.group(3).lower() != 'ac' else 'ac'}"
        ),
    ),
    # open the trunk and left door (verb ellipsis on 2nd clause)
    (
        re.compile(
            r"\b(open|close)\s+(?:the\s+)?(trunk|boot|tailgate)\s+and\s+(?:the\s+)?(left|right)\s+(door)s?\b",
            re.IGNORECASE,
        ),
        lambda m: f"{m.group(1)} {m.group(2)} {m.group(1)} {m.group(3)} {m.group(4)}",
    ),
    # turn on the ac and open window (cross-verb: ac + door/window group)
    (
        re.compile(
            r"\b(turn on|turn off)\s+(?:the\s+)?(ac|lights?)\s+and\s+(open|close)\s+(?:the\s+)?(left|right)\s+(door)s?\b",
            re.IGNORECASE,
        ),
        lambda m: (
            f"{m.group(1)} {m.group(2).rstrip('s') if m.group(2).lower() != 'ac' else 'ac'} "
            f"{m.group(3)} {m.group(4)} {m.group(5)}"
        ),
    ),
    # vi: mở/đóng cửa trái và (cửa) phải
    (
        re.compile(r"(mở|đóng)\s+cửa\s+(trái|phải)\s+và\s+(?:cửa\s+)?(trái|phải)", re.IGNORECASE),
        lambda m: f"{m.group(1)} cửa {m.group(2)} {m.group(1)} cửa {m.group(3)}",
    ),
    # vi: mở/đóng cả hai cửa (sổ) / mở hai cửa
    (
        re.compile(r"(mở|đóng)\s+(?:cả\s+hai|hai|tất\s+cả)\s+cửa", re.IGNORECASE),
        lambda m: f"{m.group(1)} cửa trái {m.group(1)} cửa phải",
    ),
]


def _expand_elliptical(text: str) -> str:
    """Expand coordinated/elliptical commands into concatenated singles."""
    out = text
    for pat, repl in _ELLIPTICAL_RES:
        out = pat.sub(repl, out)
    return out


def _parse_batch_json(text: str) -> tuple[list[tuple[str, str]], dict] | None:
    """Parse a structured batch payload: {"actions": [{"action","component"}]}.

    Returns (pairs, meta) or None when text is not a batch payload.
    """
    raw = (text or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    items = data.get("actions")
    if not isinstance(items, list):
        return None
    valid_components = {pair[0] for pair in GRAPHQL_ACTIONS}
    valid_actions = {"open", "close"}
    pairs: list[tuple[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        action = str(it.get("action", "")).strip().lower()
        component = str(it.get("component", "")).strip().lower()
        if action == "turn_on":
            action = "open"
        elif action == "turn_off":
            action = "close"
        if action in valid_actions and component in valid_components:
            pair = (component, action)
            if pair not in pairs:
                pairs.append(pair)
    return pairs, {"match_source": "batch_json", "batch_size": len(items)}


def _match_actions_with_meta(
    user_text: str,
    matcher: EmbeddingActionMatcher | None,
) -> tuple[list[tuple[str, str]], dict]:
    lower = (user_text or "").strip().lower()
    if not lower:
        return [], {"match_source": "empty"}

    # Expand coordinated/elliptical forms first so every action becomes an
    # explicit phrase: "open left and right door" -> "open left door open right door".
    expanded = _expand_elliptical(lower)

    # Phrase matching on the expanded text; keep hits ordered by their
    # position so actions execute in the order the user said them.
    hits: list[tuple[int, tuple[str, str]]] = []
    for phrases, pair in PHRASES_TO_ACTION:
        for p in phrases:
            idx = expanded.find(p)
            if idx >= 0:
                hits.append((idx, pair))
                break
    if hits:
        hits.sort(key=lambda t: t[0])
        matched: list[tuple[str, str]] = []
        for _, pair in hits:
            if pair not in matched:
                matched.append(pair)
        return matched, {
            "match_source": "elliptical" if expanded != lower else "phrase",
            "matched_label": ", ".join(
                COMPONENT_ACTION_TO_LABEL.get(pair, " ".join(pair)) for pair in matched
            ),
        }

    matched = _match_command_tokens(expanded)
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

        # Structured batch payload first ({"actions":[...]}), then text matching.
        batch = _parse_batch_json(text)
        if batch is not None:
            pairs, match_meta = batch
        else:
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
        results: list[dict] = []

        def _dispatch_one(component: str, action: str) -> bool:
            fn_name, act = GRAPHQL_ACTIONS[(component, action)]
            if not self._graphql_client:
                return True
            try:
                self._graphql_client.send_request(fn_name, {"action": act})
                return True
            except Exception as e:
                logger.warning("GraphQL failed for %s/%s: %s", component, action, e)
                return False

        # Parallel dispatch: all GraphQL mutations fire concurrently (stateless
        # HTTP) so multi-action commands cost ~1 round-trip, not N.
        import asyncio
        oks = await asyncio.gather(
            *[
                asyncio.to_thread(_dispatch_one, component, action)
                for component, action in pairs
            ]
        )

        for (component, action), ok in zip(pairs, oks):
            label = COMPONENT_ACTION_TO_LABEL.get((component, action))

            # Update local car simulator if active (PC / simulation mode)
            if ok:
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

            results.append({
                "component": component,
                "action": action,
                "label": label or f"{action} {component}",
                "ok": ok,
            })
            if ok and label:
                actions.append(label)

        match_meta["results"] = results
        response_text = json.dumps({
            "success": bool(actions),
            "message": actions,
            "meta": match_meta,
        })
        await event_queue.enqueue_event(new_agent_text_message(response_text))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")
