import os

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from langchain_api.agent.agent import Agent
from langchain_api.agent.context import AgentContext
from langchain_api.constant import root_dir
from langchain_api.endpoint import add_general_api_endpoint
from langchain_api.patch.langchain import patch_langchain

try:
    if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        # 添加可观测性组件
        from phoenix.otel import register

        tracer_provider = register(
            project_name="default",
            auto_instrument=True,
        )
except ImportError:
    pass
patch_langchain()

frontend_path = root_dir / "frontend_old"

app = FastAPI()
# 解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 将 html 路由到 /
app.mount(
    "/web",
    StaticFiles(
        directory=frontend_path,
        html=True,
    ),
    name="frontend",
)


@app.get("/")
def redirect_to_frontend():
    return RedirectResponse(url="/web/index.html")


agent = Agent(deep_agent=True).get_agent()
# 支持 AG-UI 协议
add_langgraph_fastapi_endpoint(
    app=app,
    agent=LangGraphAGUIAgent(
        name="sample_agent",
        description="An example agent to use as a starting point for your own agent.",
        graph=agent,
    ),
    path="/api/ag_ui",
)
# 添加 general_api 端点
add_general_api_endpoint(
    app=app,
    agent=agent,
    path="/api/general_api",
    context=AgentContext,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7869)
