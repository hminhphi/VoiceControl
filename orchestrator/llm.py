import logging
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

from utils import read_config

load_dotenv()

logger = logging.getLogger("orchestrator.llm")


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

        # create client once
        self.client = self._create_client()

        logger.info(
            "LlmClient ready provider=%s model=%s",
            self.provider,
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

        async for chunk in stream:

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta and delta.content:

                if first_token is None:
                    first_token = time.perf_counter() - t0

                token_count += 1
                yield delta.content

        elapsed = time.perf_counter() - t0

        logger.info(
            "LLM done provider=%s model=%s tokens=%d total=%.3fs ttft=%.3fs",
            self.provider,
            model,
            token_count,
            elapsed,
            first_token or 0.0,
        )

    async def stream_chat_completion_with_tools(self, messages, tools=None):
        """
        Stream chat completion with optional tools.
        Yields tuples:
          ("token", token_text)
          ("tool_call", tool_call_dict)
        """
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

        # llama.cpp server does not support stream with tools (returns 500 Cannot use tools with stream)
        use_stream = True
        if self.provider == "llamacpp" and tools:
            use_stream = False

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

        if not use_stream:
            resp = await self.client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - t0
            tool_calls_count = 0
            if resp.choices:
                msg = resp.choices[0].message
                if msg.content:
                    yield ("token", msg.content)
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

        stream = await self.client.chat.completions.create(**kwargs)

        tool_chunks = {}
        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta and delta.content:
                if first_token is None:
                    first_token = time.perf_counter() - t0
                token_count += 1
                yield ("token", delta.content)

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
