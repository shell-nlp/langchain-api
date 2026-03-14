from pathlib import Path

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_api.agent.agent import Agent, CustomContext
from langchain_api.endpoint import add_general_api_endpoint

root_path = Path(__file__).parent.parent

frontend_path = root_path / "frontend"

app = FastAPI()

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


class CustomContext(BaseModel):
    internet_search: bool = Field(False, description="是否允许使用互联网搜索")
    deep_thinking: bool = Field(False, description="是否启用深度思考")


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
    context=CustomContext,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7869)
