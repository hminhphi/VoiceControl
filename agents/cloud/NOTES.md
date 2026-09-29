# Cloud Agent

## 1. Folder structure

```
agents/cloud/
├── main.py              # A2A server entry point
├── agent_executor.py    # CloudAgentExecutor (Tavily web search)
├── agent_card.py        # Skills: news_search, web_search
├── prompts.py           # Orchestrator description (PURPOSE/INPUT/OUTPUT/ROUTING)
├── web_search.py        # WebSearchClient (Tavily API wrapper)
├── pyproject.toml
└── Dockerfile
```

## 2. Logic flow

(No LLM; uses keyword detection to route between news and general search.)

1. Get user text from A2A context.
2. If text empty → return `{ success: false, message: "Please send a search query..." }`.
3. **Keyword detection**: if query contains news-related words (news, latest, breaking, headlines, etc.) → call `search_news()`; otherwise → call `search_general()`.
4. `search_news()` calls Tavily with `topic="news"`, `include_answer=True`.
5. `search_general()` calls Tavily with `topic="general"`, `include_answer=True`.
6. Tavily returns `answer` (LLM-generated summary) + `results[]` (source links).
7. Format and return `{ success: true, message: answer, sources: [{title, url, snippet}] }`.

## 3. Environment variables

| Variable | Description | Example |
|---|---|---|
| `TAVILY_API_KEY` | Tavily API key | `tvly-xxxxxxxxxxxx` |
| `CLOUD_PORT` | Server port | `8005` |
| `CLOUD_BASE_URL` | Public URL | `http://localhost:8005` |

## 4. Use cases covered

| Query | Detection | Tavily topic |
|---|---|---|
| "latest AI news" | keyword "latest" + "news" | `news` |
| "Toyota company news" | keyword "news" | `news` |
| "VinFast company info" | no news keyword | `general` |
| "what is Tesla" | no news keyword | `general` |
| "breaking headlines today" | keyword "breaking" + "headlines" | `news` |
