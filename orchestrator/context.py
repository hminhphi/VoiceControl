from collections import defaultdict

MAX_CONTEXT_MESSAGES = 10


class PerAgentContext:
    def __init__(self, max_messages: int = MAX_CONTEXT_MESSAGES) -> None:
        self._max = max_messages
        self._store: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._session_recent: dict[str, list[dict[str, str]]] = defaultdict(list)

    def key(self, session_id: str, agent_id: str) -> str:
        return f"{session_id}:{agent_id}"

    def add(self, session_id: str, agent_id: str, role: str, content: str) -> None:
        k = self.key(session_id, agent_id)
        self._store[k].append({"role": role, "content": content})
        if len(self._store[k]) > self._max:
            self._store[k] = self._store[k][-self._max :]
        self._session_recent[session_id].append({"role": role, "content": content})
        if len(self._session_recent[session_id]) > self._max:
            self._session_recent[session_id] = self._session_recent[session_id][-self._max :]

    def get_recent(self, session_id: str, agent_id: str) -> list[dict[str, str]]:
        k = self.key(session_id, agent_id)
        return list(self._store[k])

    def get_recent_for_routing(self, session_id: str) -> list[dict[str, str]]:
        return list(self._session_recent[session_id])
