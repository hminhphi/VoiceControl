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
 
from router import OrchestratorRouter
from client import send_to_agents_parallel, send_to_agent
from display_graphql import send_car_manual_face_card_with_details
from llm import LlmClient
from prompts import (
    ORCHESTRATOR_PROMPT,
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
orchestrator_router = OrchestratorRouter()
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


def _format_score(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_agent_diagnostic(item: dict, route_score, route_threshold=None) -> str:
    meta = item.get("meta") or {}
    parts = [
        f"agent={item.get('agent_id')}",
        f"success={item.get('success')}",
        f"route_score={_format_score(route_score)}",
    ]
    if route_threshold is not None:
        parts.append(f"route_threshold={_format_score(route_threshold)}")

    source = meta.get("source")
    if source:
        parts.append(f"source={source}")

    if "score" in meta:
        parts.append(f"fact_score={_format_score(meta.get('score'))}")
    if "embedding_score" in meta:
        parts.append(f"embedding_score={_format_score(meta.get('embedding_score'))}")
    if meta.get("match_source"):
        parts.append(f"match_source={meta.get('match_source')}")
    if meta.get("matched_label"):
        parts.append(f"matched_label={meta.get('matched_label')!r}")
    if "local_fact_best_score" in meta:
        parts.append(f"local_fact_best_score={_format_score(meta.get('local_fact_best_score'))}")
    if "local_fact_threshold" in meta:
        parts.append(f"threshold={_format_score(meta.get('local_fact_threshold'))}")

    if "llm_rewrite_attempted" in meta:
        parts.append(f"llm_rewrite_attempted={meta.get('llm_rewrite_attempted')}")
    if "llm_rewrite_accepted" in meta:
        parts.append(f"llm_rewrite_accepted={meta.get('llm_rewrite_accepted')}")
    if "llm_rewrite_confidence" in meta and meta.get("llm_rewrite_confidence") is not None:
        parts.append(f"llm_rewrite_confidence={_format_score(meta.get('llm_rewrite_confidence'))}")
    if meta.get("llm_rewrite_query"):
        parts.append(f"llm_rewrite_query={meta.get('llm_rewrite_query')!r}")
    if meta.get("llm_rewrite_model"):
        parts.append(f"llm_rewrite_model={meta.get('llm_rewrite_model')}")
    if meta.get("llm_rewrite_error"):
        parts.append(f"llm_rewrite_error={meta.get('llm_rewrite_error')}")
    if meta.get("llm_rewrite_skipped"):
        parts.append(f"llm_rewrite_skipped={meta.get('llm_rewrite_skipped')}")

    matched_q = meta.get("matched_q")
    if matched_q:
        parts.append(f"matched_q={matched_q!r}")

    original_query = meta.get("original_query")
    rewritten_query = meta.get("rewritten_query")
    if original_query:
        parts.append(f"original_query={original_query!r}")
    if rewritten_query and rewritten_query != original_query:
        parts.append(f"rewritten_query={rewritten_query!r}")

    return " ".join(parts)


def _format_aws_face_card_diagnostic(details: dict) -> str:
    parts = [
        "aws_face_card",
        f"attempted={details.get('attempted')}",
        f"success={details.get('success')}",
        f"name={details.get('name')!r}",
    ]
    if details.get("mutation_success") is not None:
        parts.append(f"mutation_success={details.get('mutation_success')}")
    if details.get("vehicle_id"):
        parts.append(f"vehicle_id={details.get('vehicle_id')}")
    if details.get("intent_code") is not None:
        parts.append(f"intent_code={details.get('intent_code')}")
    if details.get("request_id"):
        parts.append(f"request_id={details.get('request_id')!r}")
    if details.get("status"):
        parts.append(f"status={details.get('status')!r}")
    if details.get("reason_code"):
        parts.append(f"reason_code={details.get('reason_code')!r}")
    if details.get("value"):
        parts.append(f"value={details.get('value')!r}")
    if details.get("returned_name"):
        parts.append(f"returned_name={details.get('returned_name')!r}")
    if details.get("returned_value"):
        parts.append(f"returned_value={details.get('returned_value')!r}")
    if details.get("matched_q"):
        parts.append(f"matched_q={details.get('matched_q')!r}")
    if details.get("error"):
        parts.append(f"error={details.get('error')!r}")
    if details.get("reason"):
        parts.append(f"reason={details.get('reason')!r}")
    return " ".join(parts)


def _build_llm_agent_responses(final_agent_response: dict, raw_agent_response: list[dict]) -> list[dict]:
    responses = [dict(item) for item in final_agent_response.get("responses", [])]

    raw_by_agent = {
        item.get("agent_id"): item
        for item in raw_agent_response
        if item.get("agent_id")
    }

    for response in responses:
        if response.get("agent_id") != "car_manual":
            continue

        raw = raw_by_agent.get("car_manual") or {}
        meta = raw.get("meta") or {}
        passes_threshold = meta.get("passes_threshold")
        if response.get("success") is False or passes_threshold is False:
            response["message"] = "No answer found."
            if "score" in meta:
                response["fact_score"] = meta.get("score")
            if "local_fact_threshold" in meta:
                response["threshold"] = meta.get("local_fact_threshold")
            if "passes_threshold" in meta:
                response["passes_threshold"] = passes_threshold
            continue

        documents = (raw.get("data") or {}).get("documents") or []
        best_doc = documents[0] if documents else {}

        matched_q = meta.get("matched_q") or best_doc.get("q")
        matched_answer = best_doc.get("a") or raw.get("message")

        if matched_q:
            response["matched_q"] = matched_q
        if matched_answer:
            response["matched_answer"] = matched_answer
        if meta.get("forced_query"):
            response["forced_query"] = meta.get("forced_query")
        if "score" in meta:
            response["fact_score"] = meta.get("score")
        if "local_fact_threshold" in meta:
            response["threshold"] = meta.get("local_fact_threshold")
        if "passes_threshold" in meta:
            response["passes_threshold"] = meta.get("passes_threshold")

    return responses


def _generate_natural_fallback(user_message: str, agent_responses: list[dict]) -> str:
    user_lower = (user_message or "").strip().lower()

    # Detect language
    is_ja = any(j in (user_message or "") for j in ("何時", "時間", "何日", "何曜日", "こんにちは", "おはよう", "こんばんは", "開けて", "閉めて", "つけて", "消して", "ドア", "トランク", "ライト", "エアコン", "窓", "誰"))
    is_vi = any(v in user_lower for v in ("mấy giờ", "may gio", "xem giờ", "mấy h", "ngày mấy", "thứ mấy", "chào", "xin chào", "bạn là ai", "bạn khỏe không", "mở", "đóng", "bật", "tắt", "cửa", "cốp", "đèn", "điều hòa", "máy lạnh", "kính"))

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
                return "I couldn't find relevant information in the vehicle manual."

    # 5. Any other successful response
    for item in (agent_responses or []):
        if item.get("success") is True:
            m = item.get("message")
            if isinstance(m, list):
                return " ".join(str(x) for x in m)
            if m:
                return str(m)

    return "Sorry, I can't help with that request."


async def stream_llm(session_id, messages, user_message=None, agent_responses=None):
    logger.info("Starting LLM stream for session_id=%s", session_id)
    full_text = ""

    try:
        async for token in llm_client.stream_chat_completion(messages):
            full_text += token
            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "token",
                    "content": token,
                },
            )

        logger.info("LLM stream done for session_id=%s", session_id)
        logger.info("LLM full_text for session_id=%s: %s", session_id, full_text)

        if not full_text.strip():
            if user_message is None or agent_responses is None:
                try:
                    content = messages[1]["content"]
                    if "This is the original user request:\n" in content:
                        part = content.split("This is the original user request:\n")[1]
                        user_message = part.split("\n\n")[0]
                    if "agent_response JSON:\n" in content:
                        json_part = content.split("agent_response JSON:\n")[-1]
                        data = json.loads(json_part)
                        agent_responses = data if isinstance(data, list) else data.get("responses", [])
                except Exception:
                    pass
            fallback_msg = _generate_natural_fallback(user_message, agent_responses)
            logger.info("LLM generated empty response, using fallback for session_id=%s: %s", session_id, fallback_msg)
            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "token",
                    "content": fallback_msg,
                },
            )
            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "done",
                    "message": fallback_msg,
                },
            )
        else:
            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "done",
                    "message": full_text,
                },
            )

    except Exception as e:
        logger.warning("LLM streaming unavailable (%s), falling back to natural response", e)

        if user_message is None or agent_responses is None:
            try:
                content = messages[1]["content"]
                if "This is the original user request:\n" in content:
                    part = content.split("This is the original user request:\n")[1]
                    user_message = part.split("\n\n")[0]
                if "agent_response JSON:\n" in content:
                    json_part = content.split("agent_response JSON:\n")[-1]
                    data = json.loads(json_part)
                    agent_responses = data if isinstance(data, list) else data.get("responses", [])
            except Exception:
                pass

        fallback_msg = _generate_natural_fallback(user_message, agent_responses)
        logger.info("Generated natural fallback for session_id=%s: %s", session_id, fallback_msg)

        await _send_to_session_and_frontend(
            session_id,
            {
                "type": "token",
                "content": fallback_msg,
            },
        )
        await _send_to_session_and_frontend(
            session_id,
            {
                "type": "done",
                "message": fallback_msg,
            },
        )


LOW_SCORE_FAIL_FILTER = 0.35
DIRECT_TTS_MANUAL_ANSWERS = {"ok, i will show you the fpt logo"}


def _find_direct_tts_manual_answer(agent_response: list[dict]) -> str | None:
    for item in agent_response:
        if item.get("agent_id") != "car_manual":
            continue
        if item.get("success") is not True:
            continue

        meta = item.get("meta") or {}
        if meta.get("passes_threshold") is not True:
            continue

        documents = (item.get("data") or {}).get("documents") or []
        first_doc = documents[0] if documents else {}
        answer = str(first_doc.get("a") or item.get("message") or "").strip()
        if answer.casefold() in DIRECT_TTS_MANUAL_ANSWERS:
            return answer

    return None


async def process_request(session_id, user_message, agent_ids, agent_scores):

    logger.info(
        f"Processing request session_id={session_id}, "
        f"agent_ids={agent_ids}, message='{user_message}'"
    )

    if session_id.startswith("voice-"):
        await ws_manager.send(
            "frontend",
            {"type": "user_message", "session_id": session_id, "content": user_message, "agent_ids": agent_ids},
        )

    try:

        # ---------- call agents ----------

        agent_response = await send_to_agents_parallel(
            agent_ids,
            user_message,
            context_store,
            session_id,
            registry,
        )

        logger.info(
            f"Agent responses for session_id={session_id}: {agent_response}"
        )

        if session_id.startswith("voice-"):
            for item in agent_response:
                aid = item.get("agent_id")
                await _send_to_session_and_frontend(
                    session_id,
                    {
                        "type": "session_log",
                        "content": _format_agent_diagnostic(
                            item,
                            agent_scores.get(aid),
                            orchestrator_router.threshold,
                        ),
                    },
                )

        # ---------- filter by routing score & success ----------

        filtered_agent_response = []
        for item in agent_response:
            aid = item.get("agent_id")
            success = item.get("success")
            score = agent_scores.get(aid)
            if (
                success is False
                and score is not None
                and score < LOW_SCORE_FAIL_FILTER
            ):
                logger.info(
                    "Dropping agent response due to low score and failure: "
                    "agent_id=%s, score=%.3f, success=%s",
                    aid,
                    float(score),
                    success,
                )
                continue
            filtered_agent_response.append(item)

        # ---------- best-effort AWS face-card upload from raw car_manual data ----------

        for item in filtered_agent_response:
            if item.get("agent_id") == "car_manual":
                aws_details = send_car_manual_face_card_with_details(item)
                aws_diagnostic = _format_aws_face_card_diagnostic(aws_details)
                if aws_details.get("success"):
                    logger.info(
                        "Uploaded car_manual face card to AWS for session_id=%s: %s",
                        session_id,
                        aws_diagnostic,
                    )
                else:
                    logger.warning(
                        "car_manual face card AWS upload did not succeed for session_id=%s: %s",
                        session_id,
                        aws_diagnostic,
                    )
                break

        # ---------- post process ----------

        final_agent_response = post_process_agent_response(filtered_agent_response)

        logger.info(
            f"Post-processed agent responses for session_id={session_id}: "
            f"{final_agent_response}"
        )

        # ---------- send DATA first ----------

        await _send_to_session_and_frontend(
            session_id,
            {
                "type": "data",
                "content": final_agent_response["data"],
            },
        )

        # ---------- direct TTS/manual response bypass ----------

        direct_tts_answer = _find_direct_tts_manual_answer(filtered_agent_response)
        if direct_tts_answer:
            logger.info(
                "Bypassing LLM for direct car_manual TTS response: session_id=%s answer=%r",
                session_id,
                direct_tts_answer,
            )
            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "token",
                    "content": direct_tts_answer,
                },
            )
            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "done",
                    "message": direct_tts_answer,
                },
            )
            return

        # ---------- prepare LLM ----------

        forced_manual_query = next(
            (
                (item.get("meta") or {}).get("forced_query")
                for item in filtered_agent_response
                if item.get("agent_id") == "car_manual"
            ),
            None,
        )
        llm_user_message = forced_manual_query or user_message
        llm_agent_responses = _build_llm_agent_responses(
            final_agent_response,
            filtered_agent_response,
        )

        now_str = _get_local_now().strftime("%A, %B %d, %Y %I:%M %p")

        messages = [
            {"role": "system", "content": f"{ORCHESTRATOR_PROMPT}\n\nContext:\nCurrent local time and date: {now_str}"},
            {
                "role": "user",
                "content": (
                    f"This is the original user request:\n{user_message}\n\n"
                    f"This is the matched request to answer:\n{llm_user_message}\n\n"
                    f"This is the agent_response JSON:\n"
                    f"{json.dumps(llm_agent_responses, ensure_ascii=False)}"
                ),
            },
        ]

        # ---------- stream LLM second ----------

        await stream_llm(
            session_id,
            messages,
            user_message=user_message,
            agent_responses=llm_agent_responses,
        )

    except Exception as e:

        logger.exception(
            f"Error processing request for session_id={session_id}: {e}"
        )

        await _send_to_session_and_frontend(
            session_id,
            {
                "type": "error",
                "message": str(e),
            },
        )
 
 
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

def _correct_stt_transcript(text: str) -> str:
    """Phonetically normalize car-control terms; returns corrected text."""
    original = text
    corrected = text
    for pat, repl in STT_CORRECTIONS:
        corrected = pat.sub(repl, corrected)
    if corrected != original:
        logger.info("[STT-correct] %r -> %r", original, corrected)
    return corrected


CAR_CONFIRMATIONS = {
    ("open", "left_door"): {
        "en": "The left door has been opened.",
        "vi": "Cửa bên trái đã được mở.",
        "ja": "左側のドアを開けました。",
    },
    ("close", "left_door"): {
        "en": "The left door has been closed.",
        "vi": "Cửa bên trái đã được đóng.",
        "ja": "左側のドアを閉めました。",
    },
    ("open", "right_door"): {
        "en": "The right door has been opened.",
        "vi": "Cửa bên phải đã được mở.",
        "ja": "右側のドアを開けました。",
    },
    ("close", "right_door"): {
        "en": "The right door has been closed.",
        "vi": "Cửa bên phải đã được đóng.",
        "ja": "右側のドアを閉めました。",
    },
    ("open", "trunk"): {
        "en": "The trunk has been opened.",
        "vi": "Cốp xe đã được mở.",
        "ja": "トランクを開けました。",
    },
    ("close", "trunk"): {
        "en": "The trunk has been closed.",
        "vi": "Cốp xe đã được đóng.",
        "ja": "トランクを閉めました。",
    },
    ("turn_on", "light"): {
        "en": "The headlights have been turned on.",
        "vi": "Đèn xe đã được bật.",
        "ja": "ライトを点灯しました。",
    },
    ("turn_off", "light"): {
        "en": "The headlights have been turned off.",
        "vi": "Đèn xe đã được tắt.",
        "ja": "ライトを消灯しました。",
    },
    ("turn_on", "ac"): {
        "en": "The air conditioner has been turned on.",
        "vi": "Điều hòa đã được bật.",
        "ja": "エアコンをつけました。",
    },
    ("turn_off", "ac"): {
        "en": "The air conditioner has been turned off.",
        "vi": "Điều hòa đã được tắt.",
        "ja": "エアコンを停止しました。",
    },
    ("open", "window"): {
        "en": "The windows are opened.",
        "vi": "Cửa sổ đã được mở.",
        "ja": "窓を開けました。",
    },
    ("close", "window"): {
        "en": "The windows are closed.",
        "vi": "Cửa sổ đã được đóng.",
        "ja": "窓を閉めました。",
    },
}

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

        user_lang = _normalize_lang_hint(language) or _detect_lang(user_message)

        messages = [
            {
                "role": "system",
                "content": (
                    f"{DORA_TOOL_SYSTEM_PROMPT}\n\n"
                    f"{_language_directive(user_lang)}\n\n"
                    f"Context:\nCurrent local time and date: {now_str}"
                ),
            },
            {"role": "user", "content": user_message},
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
            else:
                logger.info("Direct stream completed for session_id=%s: %s", session_id, full_text)
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": full_text})
            return

        # ---------------- Case 2: Tool call emitted ----------------
        tc = tool_calls[0]
        fn_name = tc.get("name")
        raw_args = tc.get("arguments", "{}")
        try:
            fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            fn_args = {}

        logger.info("LLM selected tool=%s args=%s for session_id=%s", fn_name, fn_args, session_id)

        # ====== Tool: control_car ======
        if fn_name == "control_car":
            action = str(fn_args.get("action", "")).strip().lower()
            component = str(fn_args.get("component", "")).strip().lower()

            cmd_label = CAR_ACTION_CMD_MAP.get((action, component))
            if not cmd_label:
                if component in ("light", "ac"):
                    act_norm = "turn_on" if action in ("open", "turn_on") else "turn_off"
                else:
                    act_norm = "open" if action in ("open", "turn_on") else "close"
                cmd_label = CAR_ACTION_CMD_MAP.get((act_norm, component), f"{action} {component}")
            else:
                act_norm = "turn_on" if action in ("open", "turn_on") and component in ("light", "ac") else ("turn_off" if action in ("close", "turn_off") and component in ("light", "ac") else action)

            await _send_to_session_and_frontend(
                session_id,
                {
                    "type": "session_log",
                    "content": f"[tool_call] control_car action={action} component={component} command={cmd_label!r}",
                },
            )

            # Call car_control agent
            agent_res = await send_to_agent("car_control", cmd_label, context_store, session_id, registry)

            # Post-process for data payload
            final_agent_response = post_process_agent_response([agent_res])
            await _send_to_session_and_frontend(
                session_id,
                {"type": "data", "content": final_agent_response["data"]},
            )

            # Fast Return Confirmation
            if agent_res.get("success") is not False:
                conf_dict = CAR_CONFIRMATIONS.get((act_norm, component), {})
                confirmation = conf_dict.get(user_lang) or conf_dict.get("en")
                if not confirmation:
                    if user_lang == "vi":
                        confirmation = f"Đã thực hiện lệnh {cmd_label}."
                    elif user_lang == "ja":
                        confirmation = f"{cmd_label}を実行しました。"
                    else:
                        confirmation = f"{cmd_label} executed successfully."
            else:
                if user_lang == "vi":
                    confirmation = "Xin lỗi, không thể thực hiện lệnh điều khiển xe này."
                elif user_lang == "ja":
                    confirmation = "申し訳ありませんが、その車両コマンドを実行できませんでした。"
                else:
                    confirmation = "Sorry, I couldn't execute that command on the vehicle."

            logger.info("Fast Return car_control for session_id=%s: %s", session_id, confirmation)
            await _send_to_session_and_frontend(session_id, {"type": "token", "content": confirmation})
            await _send_to_session_and_frontend(session_id, {"type": "done", "message": confirmation})
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

            # Direct TTS bypass if matched (e.g. FPT logo)
            direct_tts_answer = _find_direct_tts_manual_answer([agent_res])
            if direct_tts_answer:
                logger.info("Direct TTS bypass for session_id=%s: %s", session_id, direct_tts_answer)
                await _send_to_session_and_frontend(session_id, {"type": "token", "content": direct_tts_answer})
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": direct_tts_answer})
                return

            # Extract facts from docs or message
            docs = (agent_res.get("data") or {}).get("documents") or []
            meta = agent_res.get("meta") or {}
            rag_text = ""
            if docs and meta.get("passes_threshold") is not False:
                rag_text = "\n".join(f"Q: {d.get('q')}\nA: {d.get('a')}" for d in docs[:2])
            elif agent_res.get("message") and agent_res.get("message") != "No answer found." and meta.get("passes_threshold") is not False:
                rag_text = str(agent_res.get("message"))

            if not rag_text.strip():
                if user_lang == "vi":
                    no_info = "Tôi không tìm thấy thông tin này trong tài liệu hướng dẫn xe."
                elif user_lang == "ja":
                    no_info = "車両マニュアルに関連する情報が見つかりませんでした。"
                else:
                    no_info = "I couldn't find relevant information in the vehicle manual."
                logger.info("No manual facts found, sending localized message: %s", no_info)
                await _send_to_session_and_frontend(session_id, {"type": "token", "content": no_info})
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": no_info})
                return

            # Synthesize answer using streaming LLM Turn 2
            lang_instruction = (
                "Trả lời ngắn gọn bằng tiếng Việt (1-2 câu)." if user_lang == "vi"
                else ("1〜2文の簡潔な日本語で回答してください。" if user_lang == "ja"
                else "Answer concisely in English in 1-2 friendly sentences.")
            )
            turn2_messages = [
                {
                    "role": "system",
                    "content": f"{DORA_SYNTHESIS_SYSTEM_PROMPT}\n{lang_instruction}",
                },
                {
                    "role": "user",
                    "content": (
                        f"User Question: {user_message}\n"
                        f"Vehicle Manual Knowledge:\n{rag_text}\n\n"
                        f"Provide a natural, direct spoken answer to the user:"
                    ),
                },
            ]

            full_turn2_text = ""
            logger.info("Streaming Turn 2 synthesis for session_id=%s", session_id)
            async for token in llm_client.stream_chat_completion(turn2_messages):
                full_turn2_text += token
                await _send_to_session_and_frontend(
                    session_id,
                    {"type": "token", "content": token},
                )

            if not full_turn2_text.strip():
                fallback_fact = docs[0].get("a") if docs else rag_text
                await _send_to_session_and_frontend(session_id, {"type": "token", "content": fallback_fact})
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": fallback_fact})
            else:
                logger.info("Turn 2 synthesis completed for session_id=%s: %s", session_id, full_turn2_text)
                await _send_to_session_and_frontend(session_id, {"type": "done", "message": full_turn2_text})
            return

        else:
            logger.warning("Unknown tool call %r, falling back to direct completion", fn_name)
            await stream_llm(session_id, messages, user_message=user_message, agent_responses=[])

    except Exception as e:
        logger.exception("Error in process_request_tool_calling session_id=%s: %s", session_id, e)
        fallback_msg = _generate_natural_fallback(user_message, [])
        await _send_to_session_and_frontend(session_id, {"type": "token", "content": fallback_msg})
        await _send_to_session_and_frontend(session_id, {"type": "done", "message": fallback_msg})


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
 
 
