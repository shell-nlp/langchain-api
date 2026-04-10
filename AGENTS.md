# AGENTS.md

## Project Structure

- **Backend**: `langchain_api/` - FastAPI + LangGraph agent (Python 3.12)
- **Frontend**: `frontend/` - Next.js 16 (React 19) with CopilotKit UI
- **Legacy UI**: `frontend_old/` - Simple HTML/JS frontend (served at `/web`)

## Dev Commands

```bash
# Backend setup
cp .env.example .env
uv sync --dev

# Run backend (port 7869)
uv run uvicorn langchain_api.agent.main:app --reload --host 0.0.0.0 --port 7869

# Frontend
cd frontend && pnpm install
pnpm dev      # http://localhost:3000
pnpm build
pnpm lint

# Phoenix observability (optional, docker-compose)
docker-compose up -d phoenix  # http://localhost:6006
```

## Environment Variables

Required in `.env`:
- `OPENAI_API_BASE`, `OPENAI_API_KEY` - LLM endpoint
- `CHAT_MODEL_NAME` - e.g., `qwen3`
- `EMBEDDING_MODEL_NAME` - e.g., `qwen3-embedding`
- `ES_URL`, `ES_URSR`, `ES_PWD` - Elasticsearch for RAG

Optional:
- `BACKEND_TYPE` - `local_shell` (default), `store`, `sandbox`
- `TAVILY_API_KEY` - Enable web search tool
- `PG_DATABASE_URL` - Enable PostgresStore for persistent memory
- `USE_TOOL_SEARCH=True` - Enable deferred tool loading
- `USE_COPILOTKIT=True` - Enable CopilotKit middleware
- `PHOENIX_COLLECTOR_ENDPOINT` - Enable Phoenix tracing

## Architecture

### Backend Entry Point
`langchain_api/agent/main.py` - FastAPI app with three endpoints:
- `/api/ag_ui` - AG-UI protocol (LangGraphAGUIAgent)
- `/api/general_api` - Streaming general API (SSE)
- `/web/*` - Legacy HTML frontend

### Agent System
`langchain_api/agent/agent.py`:
- `Agent` class creates either DeepAgent (default) or ReactAgent
- `BusinessMiddleware` - handles `internet_search`, `deep_thinking` flags
- `DeferredToolMiddleware` - lazy tool loading when `USE_TOOL_SEARCH=True`
- Three backends: `local_shell`, `store`, `sandbox`

### Key Files
- `settings.py` - Loads `.env` via `pydantic_settings`
- `constant.py` - `workspace_path` = `.langchain_api/workspace`
- `patch/langchain.py` - Monkey-patches `add_ai_message_chunks` for streaming fix
- `retriever.py` - Elasticsearch RAG with BM25 + DenseVector
- `middleware/` - Skills loading, sandbox integration, RAG injection

## Sandbox Backend

Uses OpenSandbox config from `.sandbox.toml`:
```bash
# Start sandbox server
opensandbox-server --config .sandbox.toml
```

Sandbox requires Playwright: `playwright install --with-deps chromium`

## Testing

- No test framework configured yet (no `pytest`, `unittest` configs)
- Integration tests require `ANTHROPIC_API_KEY` set
