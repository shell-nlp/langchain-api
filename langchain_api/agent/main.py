import os

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_api.agent.agent import Agent
from langchain_api.agent.context import AgentContext
from langchain_api.constant import root_dir
from langchain_api.api import add_general_api_endpoint
from langchain_api.patch.langchain import patch_langchain

try:
    if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        from phoenix.otel import register

        tracer_provider = register(
            project_name="default",
            auto_instrument=True,
        )
except ImportError:
    pass
patch_langchain()

next_frontend_path = root_dir / "frontend" / "out"

app = FastAPI()
# 瑙ｅ喅璺ㄥ煙闂
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
agent = Agent(deep_agent=True).get_agent()
# 鏀寔 AG-UI 鍗忚
add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAGUIAgent(
        name="sample_agent",
        description="An example agent to use as a starting point for your own agent.",
        graph=agent,
    ),
    path="/api/ag_ui",
)
# 娣诲姞 general_api 绔偣
add_general_api_endpoint(
    app=app,
    agent=agent,
    path="/api/general_api",
    context=AgentContext,
)


if next_frontend_path.exists():
    app.mount(
        "/",
        StaticFiles(
            directory=next_frontend_path,
            html=True,
        ),
        name="next_frontend",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7869)
