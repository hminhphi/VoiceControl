# Car Manual Agent

## 1. Folder structure

```
agents/car_manual/
├── main.py
├── agent_executor.py    # CarManualAgentExecutor
├── agent_card.py
├── prompts.py
├── qa_store.py         # QAStore(brand), load / answer
├── data/
│   └── toyota.json     # Q&A data
├── pyproject.toml
└── Dockerfile
```

## 2. Logic flow

(No chat LLM; uses embedding model (sentence-transformers) for Q&A similarity search.)

1. Get user text from context.
2. If text empty → return { success: false, message: "Please ask a question...", is_play_sound: false, sound_path: null }.
3. Ensure QAStore loaded (brand from env, data from data/toyota.json).
4. Call store.answer(text) → answer.
5. Return { success: true, message: answer, is_play_sound: false, sound_path: null }.
