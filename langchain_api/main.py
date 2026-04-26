import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langchain_api.agent.main import create_agent_router
from langchain_api.constant import root_dir
from langchain_api.patch.langchain import patch_langchain
from langchain_api.rag.main import create_rag_router


def setup_observability() -> None:
    try:
        if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
            from phoenix.otel import register

            register(
                project_name="default",
                auto_instrument=True,
            )
    except ImportError:
        pass


def create_app() -> FastAPI:
    setup_observability()
    patch_langchain()

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(create_agent_router())
    app.include_router(create_rag_router())

    next_frontend_path = root_dir / "frontend" / "out"
    if next_frontend_path.exists():
        app.mount(
            "/",
            StaticFiles(
                directory=next_frontend_path,
                html=True,
            ),
            name="next_frontend",
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7869)
