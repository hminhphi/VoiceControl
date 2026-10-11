import logging
import json
import re

import asyncio
from datetime import datetime, timezone, timedelta

# Vietnam Timezone (UTC+7)
LOCAL_TZ = timezone(timedelta(hours=7))

def _get_local_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    except Exception:
        return datetime.now(LOCAL_TZ)


from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel
 
from client import send_to_agent
from display_graphql import send_car_manual_face_card_with_details
from llm import LlmClient
from prompts import (
    DORA_TOOL_SYSTEM_PROMPT,
    DORA_SYNTHESIS_SYSTEM_PROMPT,
    ORCHESTRATOR_TOOLS,
)
from utils import timeit

from context import PerAgentContext
from registry import AgentRegistry
from ws_manager import WebSocketManager
from agents.car_manual import post_process as car_manual_post_process
from agents.cloud import post_process as cloud_post_process
 
 
logger = logging.getLogger("orchestrator.orchestrator")
 
router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])
 
registry = AgentRegistry()
context_store = PerAgentContext()
ws_manager = WebSocketManager()
llm_client = LlmClient()



class SessionLogHandler(logging.Handler):
    def __init__(self, session_id: str, ws_manager: WebSocketManager):
        super().__init__()
        self.session_id = session_id
        self.ws_manager = ws_manager

    def emit(self, record):
        try:
            msg = self.format(record)
            content = f"INFO:{record.name}:{msg}"
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.ws_manager.send(self.session_id, {
                    "type": "session_log",
                    "content": content
                }))
            except RuntimeError:
                pass
        except Exception:
            pass


POST_PROCESSORS = {
    "car_manual": car_manual_post_process,
    "cloud": cloud_post_process,
}
 
 
class MessageRequest(BaseModel):
    message: str
    session_id: str = "default"
    language: str | None = None  # STT-detected language hint (en/ja/vi/ko/zh)
 
 
class AgentResponse(BaseModel):
    success: bool
    message: str


 
class MessageAccepted(BaseModel):
    accepted: bool
    session_id: str
    agent_ids: list[str]
    agent_scores: dict[str, float] = {}
    route_threshold: float = 0.0


async def _send_to_session_and_frontend(session_id: str, message: dict):
    await ws_manager.send(session_id, message)
    if session_id.startswith("voice-"):
        await ws_manager.send("frontend", {**message, "session_id": session_id})


def _store_assistant_reply(session_id: str, text: str) -> None:
    """Keep the assistant turn in context so follow-ups stay coherent."""
    text = (text or "").strip()
    if text:
        context_store.add(session_id, "llm_orchestrator", "assistant", text)










def _generate_natural_fallback(user_message: str, agent_responses: list[dict]) -> str:
    user_lower = (user_message or "").strip().lower()

    # Detect language
    # Script-based detection (vi diacritics / ja kana) instead of keyword lists:
    # the old lists missed most real commands and produced English fallbacks in
    # Vietnamese turns (measured 2026-10-06).
    _fallback_lang = _detect_lang(user_message or "")
    is_ja = _fallback_lang == "ja"
    is_vi = _fallback_lang == "vi"

    # 1. Time queries
    time_triggers = (
        "what time", "what's the time", "tell me the time", "current time",
        "time is this", "time is it", "mấy giờ", "may gio", "xem giờ", "mấy h",
        "何時", "今何時", "時間"
    )
    if any(t in user_lower or t in (user_message or "") for t in time_triggers):
        now = _get_local_now()
        if is_ja:
            return f"現在、{now.strftime('%H時%M分')}です。"
        if is_vi:
            return f"Bây giờ là {now.hour} giờ {now.minute:02d} phút."
        return f"It is currently {now.strftime('%I:%M %p').lstrip('0')}."

    # 2. Date queries
    date_triggers = (
        "what day", "today's date", "what is today", "what date",
        "ngày mấy", "ngay may", "thứ mấy", "thu may",
        "何日", "今日の日付", "何曜日"
    )
    if any(t in user_lower or t in (user_message or "") for t in date_triggers):
        now = _get_local_now()
        if is_ja:
            return f"本日は{now.strftime('%Y年%m月%d日')}です。"
        if is_vi:
            return f"Hôm nay là {now.strftime('ngày %d tháng %m năm %Y')}."
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    # 3. Small talk & greetings
    greetings = ("hello", "hi", "hey", "chào", "xin chào", "こんにちは", "おはよう", "こんばんは")
    if user_lower in greetings or any(user_lower.startswith(g + " ") for g in greetings) or any(g in (user_message or "") for g in ("こんにちは", "おはよう", "こんばんは")):
        if is_ja:
            return "こんにちは！ドラです。何かお手伝いしましょうか？"
        if is_vi:
            return "Xin chào! Tôi là Dora, tôi có thể giúp gì cho xe của bạn?"
        return "Hello! I am Dora, how can I help you today?"

    if any(q in user_lower or q in (user_message or "") for q in ("who are you", "what are you", "bạn là ai", "ban la ai", "誰")):
        if is_ja:
            return "私は車載AIアシスタントのドラです。"
        if is_vi:
            return "Tôi là Dora, trợ lý thông minh trên xe của bạn."
        return "I am Dora, your intelligent in-car voice assistant."

    if any(q in user_lower for q in ("how are you", "bạn khỏe không", "ban khoe khong")):
        if is_vi:
            return "Tôi rất khỏe, cảm ơn bạn! Tất cả hệ thống xe đang hoạt động bình thường."
        return "I'm doing well, thank you! All car systems are operating normally."

    # 4. Agent responses
    for item in (agent_responses or []):
        aid = item.get("agent_id")
        success = item.get("success")

        # --- Car Control ---
        if aid == "car_control":
            if success is True:
                msg = item.get("message")
                actions = msg if isinstance(msg, list) else [str(msg)]
                natural_actions = []

                action_map_ja = {
                    "open left door": "左側のドアを開けました。",
                    "close left door": "左側のドアを閉めました。",
                    "open right door": "右側のドアを開けました。",
                    "close right door": "右側のドアを閉めました。",
                    "open trunk": "トランクを開けました。",
                    "close trunk": "トランクを閉めました。",
                    "turn on light": "ライトを点灯しました。",
                    "turn off light": "ライトを消灯しました。",
                    "turn on ac": "エアコンをつけました。",
                    "turn off ac": "エアコンを停止しました。",
                    "open window": "窓を開けました。",
                    "close window": "窓を閉めました。",
                }
                action_map_vi = {
                    "open left door": "Cửa bên trái đã được mở.",
                    "close left door": "Cửa bên trái đã được đóng.",
                    "open right door": "Cửa bên phải đã được mở.",
                    "close right door": "Cửa bên phải đã được đóng.",
                    "open trunk": "Cốp xe đã được mở.",
                    "close trunk": "Cốp xe đã được đóng.",
                    "turn on light": "Đèn xe đã được bật.",
                    "turn off light": "Đèn xe đã được tắt.",
                    "turn on ac": "Điều hòa đã được bật.",
                    "turn off ac": "Điều hòa đã được tắt.",
                    "open window": "Cửa sổ đã được mở.",
                    "close window": "Cửa sổ đã được đóng.",
                }
                action_map_en = {
                    "open left door": "The left door has been opened.",
                    "close left door": "The left door has been closed.",
                    "open right door": "The right door has been opened.",
                    "close right door": "The right door has been closed.",
                    "open trunk": "The trunk has been opened.",
                    "close trunk": "The trunk has been closed.",
                    "turn on light": "The headlights have been turned on.",
                    "turn off light": "The headlights have been turned off.",
                    "turn on ac": "The air conditioner has been turned on.",
                    "turn off ac": "The air conditioner has been turned off.",
                    "open window": "The windows are opened.",
                    "close window": "The windows are closed.",
                }

                curr_map = action_map_ja if is_ja else (action_map_vi if is_vi else action_map_en)

                for a in actions:
                    a_clean = str(a).strip().lower()
                    # Also normalize variants
                    if "left door" in a_clean and "open" in a_clean: a_clean = "open left door"
                    elif "left door" in a_clean and "close" in a_clean: a_clean = "close left door"
                    elif "right door" in a_clean and "open" in a_clean: a_clean = "open right door"
                    elif "right door" in a_clean and "close" in a_clean: a_clean = "close right door"
                    elif "trunk" in a_clean and "open" in a_clean: a_clean = "open trunk"
                    elif "trunk" in a_clean and "close" in a_clean: a_clean = "close trunk"
                    elif "light" in a_clean and ("open" in a_clean or "on" in a_clean): a_clean = "turn on light"
                    elif "light" in a_clean and ("close" in a_clean or "off" in a_clean): a_clean = "turn off light"
                    elif "ac" in a_clean and ("open" in a_clean or "on" in a_clean): a_clean = "turn on ac"
                    elif "ac" in a_clean and ("close" in a_clean or "off" in a_clean): a_clean = "turn off ac"
                    elif "window" in a_clean and "open" in a_clean: a_clean = "open window"
                    elif "window" in a_clean and "close" in a_clean: a_clean = "close window"

                    if a_clean in curr_map:
                        natural_actions.append(curr_map[a_clean])
                    else:
                        natural_actions.append(action_map_en.get(a_clean, f"{a_clean} executed."))

                if natural_actions:
                    delimiter = "" if is_ja else " "
                    return delimiter.join(natural_actions)
                return "操作が完了しました。" if is_ja else ("Lệnh xe đã được thực hiện." if is_vi else "Car command executed successfully.")
            else:
                if is_ja:
                    return "申し訳ありませんが、その車両コマンドを実行できませんでした。"
                if is_vi:
                    return "Xin lỗi, không thể thực hiện lệnh điều khiển xe này."
                return "Sorry, I couldn't execute that command on the vehicle."

        # --- Car Manual ---
        elif aid == "car_manual":
            if success is True or item.get("matched_answer"):
                ans = item.get("matched_answer") or item.get("message")
                if isinstance(ans, list):
                    ans = " ".join(str(x) for x in ans)
                ans = str(ans or "").strip()
                if ans and ans != "No answer found.":
                    return ans
            if item.get("message") and item.get("message") != "No answer found.":
                return str(item.get("message")).strip()
            if any(w in user_lower for w in ("how", "what", "where", "guide", "manual", "làm sao", "ở đâu")):
                if is_ja:
                    return "車両マニュアルに関連する情報が見つかりませんでした。"
                if is_vi:
                    return "Tôi không tìm thấy thông tin này trong tài liệu hướng dẫn xe."
                return "I couldn't find relevant information in the vehicle manual."

    # 5. Any other successful response
    for item in (agent_responses or []):
        if item.get("success") is True:
            m = item.get("message")
            if isinstance(m, list):
                return " ".join(str(x) for x in m)
            if m:
                return str(m)

    if is_ja:
        return "申し訳ありませんが、そのご質問にはお答えできません。"
    if is_vi:
        return "Xin lỗi, tôi không thể giúp với yêu cầu đó."
    return "Sorry, I can't help with that request."








 
 
def _detect_lang(text: str) -> str:
    if any("\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff" for c in text):
        return "ja"
    vi_chars = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    if any(c in vi_chars for c in text.lower()):
        return "vi"
    vi_words = {"mo", "dong", "bat", "tat", "cua", "cop", "den", "kinh", "xe", "may", "gio", "hom", "nay"}
    tokens = set(text.lower().split())
    if tokens & vi_words:
        return "vi"
    return "en"


# Explicit, model-independent language directive for the active tool-calling path
# (the local LLM does not reliably follow the generic "user's language" hint).
_LANGUAGE_DIRECTIVES = {
    "vi": "Luôn trả lời bằng tiếng Việt, kể cả khi câu hỏi trộn nhiều ngôn ngữ.",
    "ja": "必ず日本語のみで回答してください。質問が他言語でも日本語で答えてください。",
    "en": "Always answer in English, even if the question mixes languages.",
}


def _language_directive(lang: str) -> str:
    return _LANGUAGE_DIRECTIVES.get(lang, _LANGUAGE_DIRECTIVES["en"])


def _normalize_lang_hint(lang) -> str | None:
    """Accept an STT language hint (en/ja/vi/ko/zh or locale) and normalize it."""
    if not lang:
        return None
    base = str(lang).strip().lower().replace("_", "-").split("-")[0]
    return base if base in ("en", "ja", "vi", "ko", "zh") else None


# ── STT phonetic auto-correction (chunk→trunk etc.) ─────────────────────
# Runs BEFORE LLM tool-calling so the model never sees the typo.
STT_CORRECTIONS: list[tuple[re.Pattern, str]] = [
    # trunk family (chunk/chank/trank/tronk common with sherpa_onnx)
    (re.compile(r"\bchunks?\b", re.IGNORECASE), "trunk"),
    (re.compile(r"\bchanks?\b", re.IGNORECASE), "trunk"),
    (re.compile(r"\btranks?\b", re.IGNORECASE), "trunk"),
    (re.compile(r"\btronks?\b", re.IGNORECASE), "trunk"),
    (re.compile(r"\btroncks?\b", re.IGNORECASE), "trunk"),
    (re.compile(r"\bcrunks?\b", re.IGNORECASE), "trunk"),
    # door
    (re.compile(r"\bdore\b", re.IGNORECASE), "door"),
    (re.compile(r"\bdoa\b", re.IGNORECASE), "door"),
    # light
    (re.compile(r"\blites?\b", re.IGNORECASE), "light"),
    (re.compile(r"\blightes\b", re.IGNORECASE), "light"),
    # window
    (re.compile(r"\bwindo\b", re.IGNORECASE), "window"),
    (re.compile(r"\bwindown\b", re.IGNORECASE), "window"),
    (re.compile(r"\bwinow\b", re.IGNORECASE), "window"),
    # ac
    (re.compile(r"\baysee\b", re.IGNORECASE), "ac"),
    (re.compile(r"\ba/c\b", re.IGNORECASE), "ac"),
]

def _turn2_lang_instruction(user_lang: str) -> str:
    if user_lang == "vi":
        return "Trả lời ngắn gọn bằng tiếng Việt (1-2 câu)."
    if user_lang == "ja":
        return "1〜2文の簡潔な日本語で回答してください。"
    return "Answer concisely in English in 1-2 friendly sentences."


async def _stream_turn2(session_id: str, turn2_messages: list) -> str:
    """Stream the LLM Turn-2 synthesis and return the full reply text."""
    full_text = ""
    async for token in llm_client.stream_chat_completion(turn2_messages):
        full_text += token
        await _send_to_session_and_frontend(
            session_id, {"type": "token", "content": token}
        )
    return full_text


def _correct_stt_transcript(text: str) -> str:
    """Phonetically normalize car-control terms; returns corrected text."""
    original = text
    corrected = text
    for pat, repl in STT_CORRECTIONS:
        corrected = pat.sub(repl, corrected)
    if corrected != original:
        logger.info("[STT-correct] %r -> %r", original, corrected)
    return corrected



CAR_ACTION_CMD_MAP = {
    ("open", "left_door"): "open left door",
    ("close", "left_door"): "close left door",
    ("open", "right_door"): "open right door",
    ("close", "right_door"): "close right door",
    ("open", "trunk"): "open trunk",
    ("close", "trunk"): "close trunk",
    ("open", "light"): "turn on light",
    ("close", "light"): "turn off light",
    ("turn_on", "light"): "turn on light",
    ("turn_off", "light"): "turn off light",
    ("open", "ac"): "turn on ac",
    ("close", "ac"): "turn off ac",
    ("turn_on", "ac"): "turn on ac",
    ("turn_off", "ac"): "turn off ac",
    ("open", "window"): "open window",
    ("close", "window"): "close window",
}

CAR_COMPONENTS = ("left_door", "right_door", "trunk", "light", "ac", "window")


def _normalize_car_actions(fn_args: dict) -> list[tuple[str, str]]:
    """Extract (action, component) pairs from a control_car tool-call args.

    Supports the batch `actions` array and the legacy single fields; dedupes
    while preserving order.
    """
    pairs: list[tuple[str, str]] = []

    def _add(action, component):
        a = str(action or "").strip().lower()
        c = str(component or "").strip().lower()
        if a in ("open", "close", "turn_on", "turn_off") and c in CAR_COMPONENTS:
            if (a, c) not in pairs:
                pairs.append((a, c))

    items = fn_args.get("actions")
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                _add(it.get("action"), it.get("component"))
    if not pairs:
        _add(fn_args.get("action"), fn_args.get("component"))
    return pairs








async def process_request_tool_calling(session_id: str, user_message: str, language: str | None = None):
    logger.info("Processing tool-calling request session_id=%s, message=%r", session_id, user_message)

    # ── STT auto-correction before any routing/LLM ──────────────────
    corrected = _correct_stt_transcript(user_message)
    if corrected != user_message:
        user_message = corrected
        logger.info("STT corrected message for session_id=%s: %r", session_id, user_message)

    if session_id.startswith("voice-"):
        await ws_manager.send(
            "frontend",
            {
                "type": "user_message",
                "session_id": session_id,
                "content": user_message,
                "agent_ids": ["llm_orchestrator"],
            },
        )

    try:
        now = _get_local_now()
        now_str = now.strftime("%A, %B %d, %Y %I:%M %p")

        # Prefer the transcript's own script (vi diacritics / ja kana) over the
        # STT hint: short/quiet speech often carries a wrong hint (2026-10-06:
        # VI speech hinted as "en" produced English fallbacks inside a VI reply).
        _script_lang = _detect_lang(user_message)
        user_lang = _script_lang if _script_lang in ("vi", "ja") else (
            _normalize_lang_hint(language) or _script_lang
        )

        # Conversation history for this session (user + assistant turns, capped by
        # PerAgentContext). The current user message was already added at ingress
        # (post_message); append it only if it is missing (defensive).
        history = [
            m
            for m in context_store.get_recent(session_id, "llm_orchestrator")
            if isinstance(m.get("content"), str)
        ]
        if not history or history[-1].get("content") != user_message:
            history = [*history, {"role": "user", "content": user_message}]

        messages = [
            {
                "role": "system",
                "content": (
                    f"{DORA_TOOL_SYSTEM_PROMPT}\n\n"
                    f"{_language_directive(user_lang)}\n\n"
                    f"Context:\nCurrent local time and date: {now_str}"
                ),
            },
            *history,
        ]

        text_tokens = []
        tool_calls = []

        async for item_type, data in llm_client.stream_chat_completion_with_tools(messages, ORCHESTRATOR_TOOLS):
            if item_type == "token":
                text_tokens.append(data)
                await _send_to_session_and_frontend(
                    session_id,
                    {"type": "token", "content": data},
                )
            elif item_type == "tool_call":
                tool_calls.append(data)

        # ---------------- Case 1: Direct streaming (No tool call) ----------------
        if not tool_calls:
            full_text = "".join(text_tokens).strip()
            if not full_text:
                fallback_msg = _generate_natural_fallback(user_message, [])
                logger.info("Direct stream empty, using natural fallback: %s", fallback_msg)
                await _send_to_session_and_frontend(session_id, {"type": "token", "content": fallback_msg})
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": fallback_msg})
                _store_assistant_reply(session_id, fallback_msg)
            else:
                logger.info("Direct stream completed for session_id=%s: %s", session_id, full_text)
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": full_text})
                _store_assistant_reply(session_id, full_text)
            return

        # ---------------- Case 2: Tool call(s) emitted ----------------
        # Execute EVERY tool call emitted (was: only tool_calls[0]); merge all
        # control_car actions into ONE batch dispatch to car_control.
        parsed_calls = []
        for tc in tool_calls:
            _fn = tc.get("name")
            raw_args = tc.get("arguments", "{}")
            try:
                _args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception:
                _args = {}
            if not isinstance(_args, dict):
                _args = {}
            parsed_calls.append((_fn, _args))
            logger.info("LLM selected tool=%s args=%s for session_id=%s", _fn, _args, session_id)

        car_pairs: list[tuple[str, str]] = []
        for _fn, _args in parsed_calls:
            if _fn == "control_car":
                for pair in _normalize_car_actions(_args):
                    if pair not in car_pairs:
                        car_pairs.append(pair)

        fn_name = "control_car" if car_pairs else (parsed_calls[0][0] if parsed_calls else "")
        fn_args = parsed_calls[0][1] if parsed_calls else {}

        # ====== Tool: control_car (batch) ======
        if fn_name == "control_car":
            cmd_labels = [CAR_ACTION_CMD_MAP.get(p, f"{p[0]} {p[1]}") for p in car_pairs]

            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "session_log",
                    "content": f"[tool_call] control_car actions={len(car_pairs)} commands={cmd_labels!r}",
                },
            )

            # ONE batch call to car_control: JSON actions payload
            batch_payload = json.dumps(
                {"actions": [{"action": a, "component": c} for a, c in car_pairs]},
                ensure_ascii=False,
            )
            agent_res = await send_to_agent(
                "car_control", user_message, context_store, session_id, registry,
                payload_override=batch_payload,
            )

            # Post-process for data payload
            final_agent_response = post_process_agent_response([agent_res])
            await _send_to_session_and_frontend(
                session_id,
                {"type": "data", "content": final_agent_response["data"]},
            )

            # Per-action results (car_control returns results[]); fall back to
            # the aggregate success flag for backward compatibility.
            results = (agent_res.get("meta") or {}).get("results") or []

            # Pure-LLM: no template fast-return. Feed the tool results back to
            # the LLM (Turn 2) so it composes the confirmation naturally and in
            # the user's language.
            if results:
                facts = json.dumps(
                    {
                        "actions": [
                            {
                                "action": r.get("action"),
                                "component": r.get("component"),
                                "ok": r.get("ok"),
                            }
                            for r in results
                        ]
                    },
                    ensure_ascii=False,
                )
            else:
                facts = json.dumps(
                    {
                        "actions": [
                            {
                                "action": a,
                                "component": c,
                                "ok": agent_res.get("success") is not False,
                            }
                            for a, c in car_pairs
                        ]
                    },
                    ensure_ascii=False,
                )

            turn2_messages = [
                {
                    "role": "system",
                    "content": (
                        f"{DORA_SYNTHESIS_SYSTEM_PROMPT}\n"
                        f"{_turn2_lang_instruction(user_lang)}"
                    ),
                },
                *history[:-1][-6:],
                {
                    "role": "user",
                    "content": (
                        f"User Request: {user_message}\n"
                        f"Vehicle command results (JSON):\n{facts}\n\n"
                        f"Confirm naturally what was done or what failed, "
                        f"in 1-2 short spoken sentences:"
                    ),
                },
            ]

            full_turn2_text = await _stream_turn2(session_id, turn2_messages)
            if not full_turn2_text.strip():
                fallback_msg = _generate_natural_fallback(user_message, [agent_res])
                await _send_to_session_and_frontend(session_id, {"type": "token", "content": fallback_msg})
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": fallback_msg})
                _store_assistant_reply(session_id, fallback_msg)
            else:
                logger.info(
                    "Turn 2 synthesis completed (car_control) for session_id=%s: %s",
                    session_id,
                    full_turn2_text,
                )
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": full_turn2_text})
                _store_assistant_reply(session_id, full_turn2_text)
            return

        # ====== Tool: search_car_manual ======
        elif fn_name == "search_car_manual":
            query = str(fn_args.get("query", user_message)).strip()

            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "session_log",
                    "content": f"[tool_call] search_car_manual query={query!r}",
                },
            )

            # Call car_manual agent
            agent_res = await send_to_agent("car_manual", query, context_store, session_id, registry)

            # AWS Face Card
            aws_details = send_car_manual_face_card_with_details(agent_res)
            if aws_details.get("success"):
                logger.info("Uploaded car_manual face card to AWS for session_id=%s", session_id)

            # Post-process and send data payload (face cards, images, URLs)
            final_agent_response = post_process_agent_response([agent_res])
            await _send_to_session_and_frontend(
                session_id,
                {"type": "data", "content": final_agent_response["data"]},
            )

            # Extract facts from docs or message
            docs = (agent_res.get("data") or {}).get("documents") or []
            meta = agent_res.get("meta") or {}
            rag_text = ""
            if docs and meta.get("passes_threshold") is not False:
                rag_text = "\n".join(f"Q: {d.get('q')}\nA: {d.get('a')}" for d in docs[:2])
            elif agent_res.get("message") and agent_res.get("message") != "No answer found." and meta.get("passes_threshold") is not False:
                rag_text = str(agent_res.get("message"))

            # Pure-LLM: always synthesize via Turn 2, even with no facts (the
            # LLM answers naturally that it could not find the information).
            knowledge = (
                rag_text if rag_text.strip()
                else "(no relevant manual information found)"
            )
            turn2_messages = [
                {
                    "role": "system",
                    "content": (
                        f"{DORA_SYNTHESIS_SYSTEM_PROMPT}\n"
                        f"{_turn2_lang_instruction(user_lang)}"
                    ),
                },
                *history[:-1][-6:],
                {
                    "role": "user",
                    "content": (
                        f"User Question: {user_message}\n"
                        f"Vehicle Manual Knowledge:\n{knowledge}\n\n"
                        f"Provide a natural, direct spoken answer to the user:"
                    ),
                },
            ]

            full_turn2_text = await _stream_turn2(session_id, turn2_messages)
            if not full_turn2_text.strip():
                fallback_msg = _generate_natural_fallback(user_message, [agent_res])
                await _send_to_session_and_frontend(session_id, {"type": "token", "content": fallback_msg})
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": fallback_msg})
                _store_assistant_reply(session_id, fallback_msg)
            else:
                logger.info("Turn 2 synthesis completed for session_id=%s: %s", session_id, full_turn2_text)
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": full_turn2_text})
                _store_assistant_reply(session_id, full_turn2_text)
            return

        else:
            logger.warning("Unknown tool call %r; using natural fallback", fn_name)
            fallback_msg = _generate_natural_fallback(user_message, [])
            await _send_to_session_and_frontend(session_id, {"type": "token", "content": fallback_msg})
            await _send_to_session_and_frontend(session_id, {"type": "done", "message": fallback_msg})
            _store_assistant_reply(session_id, fallback_msg)

    except Exception as e:
        logger.exception("Error in process_request_tool_calling session_id=%s: %s", session_id, e)
        fallback_msg = _generate_natural_fallback(user_message, [])
        await _send_to_session_and_frontend(session_id, {"type": "token", "content": fallback_msg})
        await _send_to_session_and_frontend(session_id, {"type": "done", "message": fallback_msg})
        _store_assistant_reply(session_id, fallback_msg)


@router.post("/message", response_model=MessageAccepted)
@timeit
async def post_message(req: MessageRequest):
    logger.info(f"Received message request: session_id={req.session_id}, message='{req.message}'")
    user_message = (req.message or "").strip()
    session_id = req.session_id or "default"
 
    if not user_message:
        logger.warning("No message provided in the request")
        raise HTTPException(400, "message is required")

    # STT correction at ingress (shared with tool-calling path)
    _corr = _correct_stt_transcript(user_message)
    if _corr != user_message:
        logger.info("[STT-correct@ingress] %r -> %r session_id=%s", user_message, _corr, session_id)
        user_message = _corr

    # Add context
    context_store.add(session_id, "llm_orchestrator", "user", user_message)
    logger.debug(f"Context added for session_id={session_id}, agent_id=llm_orchestrator")

    # Pure LLM Tool-Calling background task
    logger.info(f"Starting background tool-calling task for session_id={session_id}")
    asyncio.create_task(
        process_request_tool_calling(session_id, user_message, req.language)
    )
 
    return MessageAccepted(
        accepted=True,
        session_id=session_id,
        agent_ids=["llm_orchestrator"],
        agent_scores={"llm_orchestrator": 1.0},
        route_threshold=0.0,
    )
 

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    logger.info(f"WebSocket connection established for session_id={session_id}")
    await ws_manager.connect(session_id, websocket)
 
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        logger.info(f"WebSocket disconnected for session_id={session_id}")
        ws_manager.disconnect(session_id)
 
 
def post_process_agent_response(agent_response):
    logger.debug(f"Post-processing agent responses: {agent_response}")

    responses = []
    urls = []
    images = []

    for item in agent_response:

        agent_id = item.get("agent_id")

        processor = POST_PROCESSORS.get(agent_id)

        if processor:
            logger.debug(f"Applying post-processor for agent_id={agent_id}")
            item = processor(item)

        # ---- collect base fields ----

        response = {
            "agent_id": agent_id,
            "success": item.get("success"),
            "message": item.get("message"),
        }

        responses.append(response)

        # ---- collect data ----

        data = item.get("data") or {}

        url = data.get("url")
        image = data.get("image")

        if url:
            urls.append(url)

        if image:
            images.append(image)

    return {
        "responses": responses,
        "data": {
            "urls": urls,
            "images": images,
        }
    }
 
 
