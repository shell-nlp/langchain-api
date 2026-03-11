from datetime import timedelta
from pathlib import Path
import tomllib

from deepagents.backends.sandbox import (
    BaseSandbox,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from opensandbox import SandboxSync
from opensandbox.config import ConnectionConfigSync

with open(Path(__file__).parent.parent.parent / ".sandbox.toml", "rb") as f:
    config = tomllib.load(f)

DOMAIN = config["server"]["host"] + ":" + str(config["server"]["port"])


class OpenSandbox(BaseSandbox):
    """
    OpenSandbox backend for DeepAgents.
    """

    def __init__(self):
        # 1. 配置连接信息
        self.config = ConnectionConfigSync()
        self.config._DEFAULT_DOMAIN = DOMAIN

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        sandbox = SandboxSync.create(
            "gpu-server:180/opensandbox/code-interpreter:v1.0.1",
            entrypoint=["/opt/opensandbox/code-interpreter.sh"],
            env={"PYTHON_VERSION": "3.11"},
            timeout=timedelta(minutes=10),
            connection_config=self.config,
        )
        with sandbox:
            exit_code = 0
            try:
                execution = sandbox.commands.run(command)
                output = execution.logs.stdout[0].text
            except Exception as e:
                output = str(e)
                exit_code = 1
            sandbox.kill()
            return ExecuteResponse(
                output=output,
                exit_code=exit_code,
                truncated=False,
            )

    @property
    def id(self) -> str:
        return "open_sandbox"

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError("upload_files is not implemented")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError("download_files is not implemented")


if __name__ == "__main__":
    sandbox = OpenSandbox()
    print(sandbox.execute("echo 'Hello OpenSandbox!'"))
