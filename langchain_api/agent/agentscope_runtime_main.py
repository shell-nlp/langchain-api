from contextlib import asynccontextmanager

from agentscope_runtime.engine import AgentApp
from fastapi import FastAPI
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from langchain_api.agent.agent import Agent
from langchain_api.agent.context import AgentContext

agent = Agent(deep_agent=True).get_agent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services as instance variables
    # global global_short_term_memory
    # global global_long_term_memory
    # app.state.short_term_mem = InMemorySaver()
    # app.state.long_term_mem = InMemoryStore()
    # global_short_term_memory = app.state.short_term_mem
    # global_long_term_memory = app.state.long_term_mem
    try:
        yield
    finally:
        pass


agent_app = AgentApp(
    app_name="LangGraphAgent",
    app_description="A LangGraph-based research assistant",
    lifespan=lifespan,
)


@agent_app.query(framework="langgraph")
async def query_func(
    self,
    msgs,
    request: AgentRequest = None,
    **kwargs,
):
    session_id = request.session_id
    user_id = request.user_id
    print(f"Received query from user {user_id} with session {session_id}")

    async for chunk, meta_data in agent.astream(
        input={"messages": msgs, "session_id": session_id, "user_id": user_id},
        stream_mode="messages",
        config={"configurable": {"thread_id": session_id}},
        context=AgentContext(session_id=session_id, user_id=user_id),
    ):
        is_last_chunk = (
            True if getattr(chunk, "chunk_position", "") == "last" else False
        )

        yield chunk, is_last_chunk


if __name__ == "__main__":
    agent_app.run(host="0.0.0.0", port=8099)
