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
import os
from nltk import pr
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
        """Upload multiple files to the filesystem.

        Args:
            files: List of (path, content) tuples where content is bytes.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order.
        """
        responses: list[FileUploadResponse] = []
        for path, content in files:
            try:
                resolved_path = path

                # Create parent directories if needed
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                fd = os.open(resolved_path, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)

                responses.append(FileUploadResponse(path=path, error=None))
            except FileNotFoundError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
            except (ValueError, OSError) as e:
                # ValueError from _resolve_path for path traversal, OSError for other file errors
                if isinstance(e, ValueError) or "invalid" in str(e).lower():
                    responses.append(
                        FileUploadResponse(path=path, error="invalid_path")
                    )
                else:
                    # Generic error fallback
                    responses.append(
                        FileUploadResponse(path=path, error="invalid_path")
                    )

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the filesystem.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
        """
        responses: list[FileDownloadResponse] = []
        for path in paths:
            try:
                resolved_path = path
                # Use flags to optionally prevent symlink following if
                # supported by the OS
                fd = os.open(resolved_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
            except FileNotFoundError:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error="file_not_found"
                    )
                )
            except PermissionError:
                responses.append(
                    FileDownloadResponse(
                        path=path, content=None, error="permission_denied"
                    )
                )
            except IsADirectoryError:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="is_directory")
                )
            except ValueError:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )
            # Let other errors propagate
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
