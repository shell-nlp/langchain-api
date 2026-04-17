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
from opensandbox.models import WriteEntry
from opensandbox.models.execd import (
    RunCommandOpts,
)
from opensandbox.models.sandboxes import Host, Volume

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
            if timeout is None:
                timeout = 60 * 10
            try:
                # TODO 内部调用 http 沙盒的接口的超时，官方未传递 timeout 参数
                execution = self.sandbox.commands.run(
                    command,
                    opts=RunCommandOpts(
                        timeout=timedelta(seconds=timeout or self.timeout)
                    ),
                )
                output = str(execution)
                # output = execution.logs.stdout
                # if output:
                #     output = "\n".join([msg.text for msg in output])
                # else:
                #     output = ""
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
        """Upload multiple files to the filesystem.

        Args:
            files: List of (path, content) tuples where content is bytes.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """

        responses: list[FileUploadResponse] = []
        write_entries = []

        for path, content in files:
            write_entries.append(WriteEntry(path=path, data=content, mode=644))

        try:
            self.sandbox.files.write_files(write_entries)
            for path, _ in files:
                responses.append(FileUploadResponse(path=path, error=None))
        except Exception:
            for path, content in files:
                try:
                    self.sandbox.files.write_file(path=path, data=content)
                    responses.append(FileUploadResponse(path=path, error=None))
                except Exception:
                    responses.append(
                        FileUploadResponse(path=path, error="unknown_error")
                    )

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the filesystem.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
        """
        # TODO 上传和下载的异常处理存在问题，暂未处理
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self.sandbox.files.read_bytes(path)
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except Exception:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="unknown_error")
                )

        return responses


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
