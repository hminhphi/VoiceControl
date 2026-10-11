import logging
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from utils import read_config

load_dotenv()

logger = logging.getLogger("orchestrator.llm")


_THINK_OPEN = '<think>'
_THINK_CLOSE = '</think>'


class _ThinkStripper:
    """Remove  thinking... blocks from a streamed chat reply.

    Some local models (Qwen3 family via llama.cpp / LM Studio) emit their
    reasoning inline in `content`. That text must never reach TTS. Handles
    tags split across streamed chunks and an unclosed block at EOF.
    """

    def __init__(self):
        self._carry = ""
        self._inside = False

    @staticmethod
    def _partial_tag_len(buf: str, tag: str) -> int:
        max_keep = min(len(buf), len(tag) - 1)
        for k in range(max_keep, 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0

    def feed(self, text: str) -> str:
        buf = self._carry + (text or "")
        self._carry = ""
        out = []
        i = 0
        while i < len(buf):
            if not self._inside:
                j = buf.find(_THINK_OPEN, i)
                if j == -1:
                    keep = self._partial_tag_len(buf[i:], _THINK_OPEN)
                    out.append(buf[i:len(buf) - keep])
                    self._carry = buf[len(buf) - keep:]
                    i = len(buf)
                else:
                    out.append(buf[i:j])
                    i = j + len(_THINK_OPEN)
                    self._inside = True
            else:
                j = buf.find(_THINK_CLOSE, i)
                if j == -1:
                    keep = self._partial_tag_len(buf[i:], _THINK_CLOSE)
                    self._carry = buf[len(buf) - keep:]
                    i = len(buf)
                else:
                    i = j + len(_THINK_CLOSE)
                    self._inside = False
        return "".join(out)

    def flush(self) -> str:
        carry = self._carry
        self._carry = ""
        return "" if self._inside else carry


def _split_for_responses(messages):
    """Responses API: system -> `instructions`, con lai -> `input` items."""
    instructions = []
    input_items = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if content:
                instructions.append(str(content))
        else:
            input_items.append({"role": role, "content": content or ""})
    return ("\n\n".join(instructions) or None), input_items


def _flatten_tools(tools):
    """Chat Completions dung {type,function:{...}}; Responses API dung phang
    {type,name,description,parameters}."""
    out = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if fn:
            out.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        else:
            out.append(t)  # da o dang phang
    return out


class LlmClient:
    """
    Unified client for:
      - OpenAI
      - llama.cpp (OpenAI-compatible server)

    Only difference = how AsyncOpenAI is created
    """

    def __init__(self, config_path="config/common_config.yaml"):

        self.config = read_config(path=config_path)["llm"]

        self.provider = self.config.get("provider", "openai").lower()
        self.model = (
            os.getenv("LOCAL_LLM_MODEL", "").strip()
            or self.config.get("model", "gpt-4o-mini")
        )

        self.temperature = float(self.config.get("temperature", 0.7))
        self.top_p = float(self.config.get("top_p", 0.9))
        self.max_tokens = int(self.config.get("max_tokens", 512))
        self.timeout = float(self.config.get("timeout", 120.0))

        # ---------- OpenAI ----------
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

        # ---------- llama.cpp ----------
        llamacpp_cfg = self.config.get("llamacpp", {})

        self.llamacpp_base_url = (
            os.getenv("LOCAL_LLM_URL", "").strip()
            or llamacpp_cfg.get("base_url", "http://host.docker.internal:8080/v1")
        )

        self.llamacpp_api_key = llamacpp_cfg.get(
            "api_key",
            "none",
        )

        # ---------- API flavour ----------
        # chat      = OpenAI-compatible /v1/chat/completions (llama.cpp / Jetson)
        # responses = OpenAI Responses API /v1/responses (x86 server: tach rieng
        #             reasoning, tool format phang, co `instructions`)
        self.api = (
            os.getenv("LLM_API", "").strip().lower()
            or str(self.config.get("api", "chat")).strip().lower()
        )

        # Bat sau lan dau server tu choi stream+tools (xem
        # stream_chat_completion_with_tools).
        self._stream_tools_blocked = False

        # create client once
        self.client = self._create_client()

        logger.info(
            "LlmClient ready provider=%s api=%s model=%s",
            self.provider,
            self.api,
            self.model,
        )

    # =====================================================
    # CLIENT FACTORY
    # =====================================================

    def _create_client(self):

        if self.provider == "openai":

            return AsyncOpenAI(
                api_key=self.openai_api_key,
                timeout=self.timeout,
            )

        elif self.provider == "llamacpp":

            return AsyncOpenAI(
                base_url=self.llamacpp_base_url,
                api_key=self.llamacpp_api_key,
                timeout=self.timeout,
            )

        else:
            raise ValueError(
                "provider must be openai or llamacpp"
            )

    # =====================================================
    # PUBLIC API
    # =====================================================

    async def stream_chat_completion(self, messages):

        if self.api == "responses":
            async for token in self._stream_responses_text(messages):
                yield token
            return

        t0 = time.perf_counter()
        first_token = None
        token_count = 0

        model = self.model
        if self.provider == "llamacpp":
            try:
                models_resp = await self.client.models.list()
                if models_resp and models_resp.data:
                    loaded_ids = [m.id for m in models_resp.data if m.id]
                    if loaded_ids and model not in loaded_ids:
                        model = loaded_ids[0]
            except Exception:
                pass

        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        stripper = _ThinkStripper()
        async for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta and delta.content:
                text = stripper.feed(delta.content)
                if text:
                    if first_token is None:
                        first_token = time.perf_counter() - t0

                    token_count += 1
                    yield text

        tail = stripper.flush()
        if tail:
            yield tail

        elapsed = time.perf_counter() - t0

        logger.info(
            "LLM done provider=%s model=%s tokens=%d total=%.3fs ttft=%.3fs",
            self.provider,
            model,
            token_count,
            elapsed,
            first_token or 0.0,
        )

    async def _stream_responses_text(self, messages):
        """Responses API, chi text (khong tools) — dung cho cau tra loi cuoi."""
        t0 = time.perf_counter()
        first_token = None
        token_count = 0
        instructions, input_items = _split_for_responses(messages)

        stream = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            stream=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
        )

        async for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    if first_token is None:
                        first_token = time.perf_counter() - t0
                    token_count += 1
                    yield delta

        logger.info(
            "LLM responses stream done provider=%s model=%s tokens=%d total=%.3fs ttft=%.3fs",
            self.provider,
            self.model,
            token_count,
            time.perf_counter() - t0,
            first_token or 0.0,
        )

    async def _responses_with_tools(self, messages, tools):
        """Responses API voi tools (non-stream): tra ve text + function_call."""
        t0 = time.perf_counter()
        instructions, input_items = _split_for_responses(messages)
        tool_calls_count = 0

        resp = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=_flatten_tools(tools),
            tool_choice="auto",
            max_output_tokens=self.max_tokens,
        )

        for item in getattr(resp, "output", None) or []:
            itype = getattr(item, "type", "")
            if itype == "function_call":
                tool_calls_count += 1
                yield ("tool_call", {
                    "id": getattr(item, "call_id", None) or getattr(item, "id", "") or "",
                    "name": getattr(item, "name", "") or "",
                    "arguments": getattr(item, "arguments", "") or "{}",
                })
            elif itype == "message":
                for part in getattr(item, "content", None) or []:
                    text = getattr(part, "text", "") or ""
                    if text:
                        yield ("token", text)

        logger.info(
            "LLM responses tool call done provider=%s model=%s tool_calls=%d total=%.3fs",
            self.provider,
            self.model,
            tool_calls_count,
            time.perf_counter() - t0,
        )

    async def stream_chat_completion_with_tools(self, messages, tools=None):
        """
        Stream chat completion with optional tools.
        Yields tuples:
          ("token", token_text)
          ("tool_call", tool_call_dict)
        """
        if self.api == "responses":
            async for item in self._responses_with_tools(messages, tools):
                yield item
            return

        t0 = time.perf_counter()
        first_token = None
        token_count = 0

        model = self.model
        if self.provider == "llamacpp":
            try:
                models_resp = await self.client.models.list()
                if models_resp and models_resp.data:
                    loaded_ids = [m.id for m in models_resp.data if m.id]
                    if loaded_ids and model not in loaded_ids:
                        model = loaded_ids[0]
            except Exception:
                pass

        # Streaming tool calls: current servers stream `tool_calls` deltas fine
        # (verified: 200 + SSE with tool_call chunks, finish_reason=tool_calls).
        # Older llama.cpp builds answered 500 "Cannot use tools with stream", so
        # try streaming first and fall back to non-streaming once if refused.
        use_stream = not (tools and self._stream_tools_blocked)

        kwargs = {
            "model": model,
            "messages": messages,
            "stream": use_stream,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        stream = None
        if use_stream:
            try:
                stream = await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if not tools:
                    raise
                logger.warning(
                    "stream+tools rejected by server (%s); retrying without stream",
                    str(e)[:120],
                )
                self._stream_tools_blocked = True

        if stream is None:
            kwargs["stream"] = False
            resp = await self.client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            tool_calls_count = 0
            if resp.choices:
                msg = resp.choices[0].message
                if msg.content:
                    _strip = _ThinkStripper()
                    text = _strip.feed(msg.content) + _strip.flush()
                    if text:
                        yield ("token", text)
                if msg.tool_calls:
                    tool_calls_count = len(msg.tool_calls)
                    for idx, tc in enumerate(msg.tool_calls):
                        yield ("tool_call", {
                            "id": tc.id or f"call_{idx}",
                            "name": tc.function.name if tc.function else "",
                            "arguments": tc.function.arguments if tc.function else "{}",
                        })
            logger.info(
                "LLM non-stream tool call done provider=%s model=%s tool_calls=%d total=%.3fs",
                self.provider,
                model,
                tool_calls_count,
                elapsed,
            )
            return

        tool_chunks = {}
        stripper = _ThinkStripper()
        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta and delta.content:
                text = stripper.feed(delta.content)
                if text:
                    if first_token is None:
                        first_token = time.perf_counter() - t0
                    token_count += 1
                    yield ("token", text)

            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    if idx not in tool_chunks:
                        tool_chunks[idx] = {"id": tc.id or f"call_{idx}", "name": "", "arguments": ""}
                    if tc.id:
                        tool_chunks[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_chunks[idx]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_chunks[idx]["arguments"] += tc.function.arguments

        elapsed = time.perf_counter() - t0

        tail = stripper.flush()
        if tail:
            yield ("token", tail)

        if tool_chunks:
            for idx in sorted(tool_chunks.keys()):
                yield ("tool_call", tool_chunks[idx])

        logger.info(
            "LLM tool stream done provider=%s model=%s tokens=%d tool_calls=%d total=%.3fs ttft=%.3fs",
            self.provider,
            model,
            token_count,
            len(tool_chunks),
            elapsed,
            first_token or 0.0,
        )


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    import asyncio

    ORCHESTRATOR_PROMPT = (
        "You are an in-car assistant. Read the user request and agent response, "
        "then reply naturally. Be concise (1-3 sentences), friendly, and reply "
        "in the same language as the user."
    )

    user_message = "Cây Cần Thăng là gì?"

    agent_response = {
        "responses": [
            {"agent_id": "car_control", "success": False, "message": ""},
            {"agent_id": "cloud", "success": True, "message": "Cây Cần Thăng là bonsai tượng trưng thăng tiến"},
        ]
    }

    messages = [
        {"role": "system", "content": ORCHESTRATOR_PROMPT},
        {
            "role": "user",
            "content": (
                f"This is the user request:\n{user_message}\n\n"
                f"This is the agent_response:\n{agent_response}"
            ),
        },
    ]

    async def main():

        client = LlmClient()

        async for t in client.stream_chat_completion(messages):
            print(t, end="", flush=True)

        print()

    asyncio.run(main())
