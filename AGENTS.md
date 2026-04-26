# AGENTS.md

## 项目结构

- **后端**：`langchain_api/`，FastAPI + LangGraph/DeepAgents，Python 3.12。
- **API 层**：`langchain_api/api/`，放置 FastAPI 路由注册和协议适配代码。
- **Agent 层**：`langchain_api/agent/`，放置 Agent 构建、上下文、状态和 Agent Router。
- **RAG 层**：`langchain_api/rag/`，放置 Elasticsearch 检索、向量图 RAG、RAG Router 和上下文。
- **中间件**：`langchain_api/middleware/`，放置业务开关、RAG 注入、工具搜索和沙箱相关中间件。
- **工具**：`langchain_api/tools/`，放置天气、网页抓取、定时任务等工具。
- **前端**：`frontend/`，Next.js + React。构建后通过 FastAPI 的 `/` 路径直接提供静态页面。
- **工作区**：`.langchain_api/workspace`，由 `langchain_api/constant.py` 定义。

## 常用命令

```bash
# 后端初始化
cp .env.example .env
uv sync --dev

# 启动统一主服务，同时包含 Agent 和 RAG 接口
uv run uvicorn langchain_api.main:app --reload --host 0.0.0.0 --port 7869

# 前端开发
cd frontend
pnpm install
pnpm dev      # http://localhost:3000
pnpm lint

# 构建 Next.js 静态前端，输出到 frontend/out
pnpm build

# Phoenix 可观测性，可选
docker-compose up -d phoenix  # http://localhost:6006
```

## 前后端入口

`langchain_api/main.py` 是唯一 FastAPI 启动入口：

- `/`：挂载 `frontend/out`，用于访问 Next.js 静态构建结果。
- `/api/agent/ag_ui`：Agent 的 AG-UI 协议接口，使用 `LangGraphAGUIAgent`。
- `/api/agent/general_api`：Agent 通用流式接口，SSE 输出。
- `/api/rag/general_api`：RAG 专用流式接口，SSE 输出。

主入口只负责创建 app、加载中间件、包含各模块 router 和挂载前端静态文件。

注意：修改前端后，需要在 `frontend/` 下执行 `pnpm build`，后端才会通过 `/` 提供最新静态页面。

## 环境变量

`.env` 必填：

- `OPENAI_API_BASE`、`OPENAI_API_KEY`：LLM 接口地址和密钥。
- `CHAT_MODEL_NAME`：聊天模型名称，例如 `qwen3`。
- `EMBEDDING_MODEL_NAME`：向量模型名称，例如 `qwen3-embedding`。
- `ES_URL`、`ES_URSR`、`ES_PWD`：Elasticsearch 连接配置。

可选：

- `BACKEND_TYPE`：`local_shell`、`store`、`sandbox`，默认 `local_shell`。
- `TAVILY_API_KEY`：启用联网搜索工具。
- `PG_DATABASE_URL`：启用 PostgresStore 持久化记忆。
- `USE_TOOL_SEARCH=True`：启用延迟工具加载。
- `USE_COPILOTKIT=True`：启用 CopilotKit 中间件。
- `PHOENIX_COLLECTOR_ENDPOINT`：启用 Phoenix tracing。

## 后端架构

### Agent 系统

`langchain_api/agent/agent.py`：

- `Agent` 默认创建 DeepAgent，也支持 ReactAgent。
- `BusinessMiddleware` 处理 `internet_search`、`deep_thinking` 等业务开关。
- `DeferredToolMiddleware` 在 `USE_TOOL_SEARCH=True` 时延迟加载工具。
- 支持 `local_shell`、`store`、`sandbox` 三种后端执行方式。

### API 层

- `langchain_api/api/endpoints.py`：通用 SSE 接口注册逻辑，具体路径由调用方传入。
- `langchain_api/api/__init__.py`：导出 API 层公共入口。


- `langchain_api/agent/main.py`：创建 Agent Router，不提供独立 app 或 uvicorn 启动入口。
- `langchain_api/rag/main.py`：创建 RAG Router，不提供独立 app 或 uvicorn 启动入口。
- `langchain_api/rag/service.py`：创建 RAG agent，并把 RAG 接口注册到 FastAPI app/router。
- `langchain_api/rag/retriever.py`：基础 Elasticsearch 检索工具，包含 DenseVector/BM25 和图 RAG 工具入口。
- `langchain_api/rag/elastic_utils.py`：Elasticsearch 基础封装，包含普通检索、向量检索和向量图检索。
- `langchain_api/rag/elastic_graph_rag.py`：基于 ES 的向量图 RAG。
  - `ElasticGraphRAG.add_texts()`：文本入库并构建图索引。
  - `ElasticGraphRAG.add_documents()`：`Document` 入库并构建图索引。
  - `ElasticGraphRAG.retrieve()`：执行实体/关系召回、图扩展、关系裁剪和 passage 回收。
  - `ElasticGraphRAG.delete_documents()`：按文档 ID 删除 passage，并清理孤立实体/关系。
  - `ElasticGraphRAG.delete_graph()`：删除当前图的三类索引。

### 关键文件

- `langchain_api/settings.py`：通过 `pydantic_settings` 加载 `.env`。
- `langchain_api/utils.py`：创建聊天模型和 embedding 模型。
- `langchain_api/constant.py`：定义根目录和工作区路径。
- `langchain_api/patch/langchain.py`：修补 LangChain 流式消息合并逻辑。

## Sandbox 后端

使用 `.sandbox.toml` 配置 OpenSandbox：

```bash
opensandbox-server --config .sandbox.toml
```

Sandbox 依赖 Playwright：

```bash
playwright install --with-deps chromium
```

## 开发注意事项

- 修改 Python 文件后，优先运行：

```bash
uv run python -m py_compile <changed_file.py>
```

- 修改前端后，优先运行：

```bash
cd frontend
pnpm lint
pnpm build
```

- 当前没有统一测试框架配置，不要随意新增测试框架或重构无关模块。
- 修改已有代码时保持最小改动，优先修根因，不要顺手改无关问题。
