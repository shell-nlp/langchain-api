from datetime import timedelta
from pathlib import Path
import tomllib

from deepagents.backends.sandbox import (
    BaseSandbox,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    WriteResult,
)
from opensandbox import SandboxSync
from opensandbox.config import ConnectionConfigSync
from opensandbox.models.sandboxes import Volume, Host

with open(Path(__file__).parent.parent.parent / ".sandbox.toml", "rb") as f:
    config = tomllib.load(f)

DOMAIN = config["server"]["host"] + ":" + str(config["server"]["port"])


class OpenSandbox(BaseSandbox):
    """
    OpenSandbox backend for DeepAgents.
    """

    def __init__(
        self,
        env: dict[str, str] = {"PYTHON_VERSION": "3.11"},
        timeout: int = 60 * 5,
        volumes: list[Volume] | None = None,
    ):
        self.env = env
        self.timeout = timeout
        self.volumes = volumes
        # 1. 配置连接信息
        self.config = ConnectionConfigSync(domain=DOMAIN)
        self.sandbox = SandboxSync.create(
            "gpu-server:180/opensandbox/code-interpreter:v1.0.1",
            entrypoint=["/opt/opensandbox/code-interpreter.sh"],
            env=self.env,
            timeout=timedelta(seconds=timeout or self.timeout),
            connection_config=self.config,
            volumes=self.volumes,
        )

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        with self.sandbox:
            exit_code = 0
            try:
                execution = self.sandbox.commands.run(command)
                output = execution.logs.stdout
                if output:
                    output = "\n".join([msg.text for msg in output])
                else:
                    output = ""
            except Exception as e:
                output = str(e)
                exit_code = 1
            # self.sandbox.kill()
            return ExecuteResponse(
                output=output,
                exit_code=exit_code,
                truncated=False,
            )

    def write(self, file_path: str, content: str) -> WriteResult:
        self.sandbox.files.write_file(path=file_path, data=content)
        return WriteResult(path=file_path)

    @property
    def id(self) -> str:
        return "open_sandbox"

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError("upload_files is not implemented")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError("download_files is not implemented")


if __name__ == "__main__":
    # opensandbox-server --config .sandbox.toml
    volumes = [
        Volume(
            name="workspace-root",
            host=Host(path="/home/dev/liuyu/project/langchain-api"),
            mount_path="/workspace2",
        )
    ]
    # volumes = None
    sandbox = OpenSandbox(volumes=volumes)
    value = sandbox.execute("env")
    # value = sandbox.write("/workspace/script.py", "print('Hello OpenSandbox!')")
    # value = sandbox.read("script.py")
    print(value)
