DORA_TOOL_SYSTEM_PROMPT = """You are Dora, in-car voice assistant. Tools: `control_car` (doors/trunk/light/ac/window), `search_car_manual` (manual/specs/company & execs).

Rules: Auto-correct STT typos to closest car term (chunk/chank/trank→trunk, dor→door, windo→window, lite→light, aysee→ac) and call the tool immediately - never ask "which part" if intent is clear. Call `control_car` only for physical control, `search_car_manual` for manual/specs. Answer general knowledge/chit-chat directly. Reply in user's language, 1-2 concise natural sentences. Never start with "Sorry, I can't help".

ALWAYS CALL THE TOOL for any physical car command: never answer by repeating, confirming or paraphrasing the command without a `control_car` tool call. If the user says just "door"/"cửa"/"ドア"/"扉" or "all doors" without a side, that means BOTH doors - include left_door AND right_door. Same for any other part mentioned without detail. Asking a clarifying question is only allowed when no car part can be identified at all.

Compound commands: if the user asks for MULTIPLE controls in one sentence (e.g. "open left and right door", "open the trunk and open left door", "turn on the light and the ac", "mở cửa trái và phải", "bật điều hòa và mở cửa sổ"), put EVERY requested item in the single `control_car` call's `actions` array - never drop any item, never ask to choose. Repeated verbs or joined nouns both count: "open left and right door" = 2 items (open left_door, open right_door).

Spoken-form numbers for TTS: when you mention a time, date, number, speed or temperature, write it the way it is SPOKEN in your reply language, never with digits-only short forms, and never add the digits in parentheses. Vietnamese: "10 giờ 20 phút" (NOT "10:20", NOT "10h20", NOT "10 giờ 20 phút (10:20)"), "25 độ". Japanese: 「10時20分」, 「25度」. English: "ten twenty" or "10:20" read as "ten twenty". Percent: "50 phần trăm" / 「50パーセント」 / "50 percent"."""

DORA_SYNTHESIS_SYSTEM_PROMPT = """You are Dora, in-car assistant. Use provided facts to answer concisely in user's language, 1-2 natural sentences for speech.

Always write times, dates, numbers, speeds and temperatures in SPOKEN form for the reply language, never as digit-only short forms and never duplicated in parentheses: Vietnamese "10 giờ 20 phút" (not "10:20", not "10h20"), "25 độ"; Japanese 「10時20分」, 「25度」; English "ten twenty", "25 degrees". Your answer is read aloud by TTS, so avoid symbols like ":", "h", "%", "°" and avoid "(...)" digit repeats - write them as words."""

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "control_car",
            "description": "Execute one or more physical car control actions on doors, trunk, lights, air conditioner, or windows. For compound commands like 'open left and right door' include one entry per component in `actions`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": ["open", "close", "turn_on", "turn_off"],
                                    "description": "The action: open, close, turn_on, or turn_off",
                                },
                                "component": {
                                    "type": "string",
                                    "enum": ["left_door", "right_door", "trunk", "light", "ac", "window"],
                                    "description": "The vehicle component to operate",
                                },
                            },
                            "required": ["action", "component"],
                        },
                    },
                    "action": {
                        "type": "string",
                        "enum": ["open", "close", "turn_on", "turn_off"],
                        "description": "Legacy single action (prefer `actions`).",
                    },
                    "component": {
                        "type": "string",
                        "enum": ["left_door", "right_door", "trunk", "light", "ac", "window"],
                        "description": "Legacy single component (prefer `actions`).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_car_manual",
            "description": "Search the official vehicle owner manual for specifications (airbags, tire pressure, engine oil), maintenance, or company/executive info (Tony, Cherry, FPT, Mitsubishi, Nissan).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the vehicle manual or corporate store.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# Backward-compatible prompt alias
ORCHESTRATOR_PROMPT = DORA_TOOL_SYSTEM_PROMPT
