import base64
import concurrent.futures
from pathlib import Path
from typing import Annotated, Literal, Optional

from deepagents.backends.utils import (
    format_grep_matches,
    truncate_if_too_long,
    validate_path,
)
from deepagents.middleware.filesystem import (
    DEFAULT_READ_LIMIT,
    DEFAULT_READ_OFFSET,
    EDIT_FILE_TOOL_DESCRIPTION,
    EXECUTE_TOOL_DESCRIPTION,
    EditResult,
    GLOB_TOOL_DESCRIPTION,
    IMAGE_EXTENSIONS,
    IMAGE_MEDIA_TYPES,
    LIST_FILES_TOOL_DESCRIPTION,
    NUM_CHARS_PER_TOKEN,
    READ_FILE_TOOL_DESCRIPTION,
    READ_FILE_TRUNCATION_MSG,
    WRITE_FILE_TOOL_DESCRIPTION,
    WriteResult,
    create_image_block,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command
from opensandbox.models.sandboxes import Host, Volume

from langchain_api.sandbox.open_sandbox import OpenSandbox

workspace_path = ""


@tool("ls", description=LIST_FILES_TOOL_DESCRIPTION)
def ls_tool(
    path: Annotated[
        str, "Absolute path to the directory to list. Must be absolute, not relative."
    ],
):
    """列出当前目录下的文件"""
    try:
        validated_path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    backend = OpenSandbox(
        volumes=[
            Volume(
                name="workspace-root",
                host=Host(
                    path="/home/dev/liuyu/project/langchain-api/.langchain_api/workspace"
                ),
                mount_path="/workspace",
            )
        ]
    )
    infos = backend.ls_info(path=validated_path)
    paths = [fi.get("path", "") for fi in infos]
    result = truncate_if_too_long(paths)
    backend.sandbox.kill()
    return str(result)


@tool("execute", description=EXECUTE_TOOL_DESCRIPTION)
def execute_tool(
    command: Annotated[str, "Shell command to execute in the sandbox environment."],
    runtime: ToolRuntime,
    timeout: Annotated[
        int | None,
        "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
    ] = None,
):
    """执行命令并返回结果"""
    resolved_backend = OpenSandbox(
        volumes=[
            Volume(
                name="workspace-root",
                host=Host(
                    path="/home/dev/liuyu/project/langchain-api/.langchain_api/workspace"
                ),
                mount_path="/workspace",
            )
        ]
    )
    result = resolved_backend.execute(command)
    # Format output for LLM consumption
    parts = [result.output]

    if result.exit_code is not None:
        status = "succeeded" if result.exit_code == 0 else "failed"
        parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

    if result.truncated:
        parts.append("\n[Output was truncated due to size limits]")
    resolved_backend.sandbox.kill()
    return "".join(parts)


token_limit = 20000


@tool("read_file", description=READ_FILE_TOOL_DESCRIPTION)
def read_file(
    file_path: Annotated[
        str, "Absolute path to the file to read. Must be absolute, not relative."
    ],
    runtime: ToolRuntime,
    offset: Annotated[
        int,
        "Line number to start reading from (0-indexed). Use for pagination of large files.",
    ] = DEFAULT_READ_OFFSET,
    limit: Annotated[
        int, "Maximum number of lines to read. Use for pagination of large files."
    ] = DEFAULT_READ_LIMIT,
):
    """读取文件内容"""
    try:
        validated_path = validate_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    backend = OpenSandbox(
        volumes=[
            Volume(
                name="workspace-root",
                host=Host(
                    path="/home/dev/liuyu/project/langchain-api/.langchain_api/workspace"
                ),
                mount_path="/workspace",
            )
        ]
    )
    ext = Path(validated_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        responses = backend.download_files([validated_path])
        if responses and responses[0].content is not None:
            media_type = IMAGE_MEDIA_TYPES.get(ext, "image/png")
            image_b64 = base64.standard_b64encode(responses[0].content).decode("utf-8")
            backend.sandbox.kill()
            return ToolMessage(
                content_blocks=[
                    create_image_block(base64=image_b64, mime_type=media_type)
                ],
                name="read_file",
                tool_call_id=runtime.tool_call_id,
                additional_kwargs={
                    "read_file_path": validated_path,
                    "read_file_media_type": media_type,
                },
            )
        if responses and responses[0].error:
            return f"Error reading image: {responses[0].error}"
        return "Error reading image: unknown error"

    result = backend.read(validated_path, offset=offset, limit=limit)

    lines = result.splitlines(keepends=True)
    if len(lines) > limit:
        lines = lines[:limit]
        result = "".join(lines)

    # Check if result exceeds token threshold and truncate if necessary
    if token_limit and len(result) >= NUM_CHARS_PER_TOKEN * token_limit:
        # Calculate truncation message length to ensure final result stays under threshold
        truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=validated_path)
        max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
        result = result[:max_content_length]
        result += truncation_msg
    backend.sandbox.kill()
    return result


@tool("write_file", description=WRITE_FILE_TOOL_DESCRIPTION)
def write_file(
    file_path: Annotated[
        str,
        "Absolute path where the file should be created. Must be absolute, not relative.",
    ],
    content: Annotated[
        str, "The text content to write to the file. This parameter is required."
    ],
    runtime: ToolRuntime,
):
    """写入文件内容"""
    backend = OpenSandbox(
        volumes=[
            Volume(
                name="workspace-root",
                host=Host(
                    path="/home/dev/liuyu/project/langchain-api/.langchain_api/workspace"
                ),
                mount_path="/workspace",
            )
        ]
    )
    try:
        validated_path = validate_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    res: WriteResult = backend.write(validated_path, content)
    if res.error:
        return res.error
    # If backend returns state update, wrap into Command with ToolMessage
    backend.sandbox.kill()
    if res.files_update is not None:
        return Command(
            update={
                "files": res.files_update,
                "messages": [
                    ToolMessage(
                        content=f"Updated file {res.path}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    return f"Updated file {res.path}"


@tool("edit_file", description=EDIT_FILE_TOOL_DESCRIPTION)
def edit_file(
    file_path: Annotated[
        str, "Absolute path to the file to edit. Must be absolute, not relative."
    ],
    old_string: Annotated[
        str,
        "The exact text to find and replace. Must be unique in the file unless replace_all is True.",
    ],
    new_string: Annotated[
        str, "The text to replace old_string with. Must be different from old_string."
    ],
    runtime: ToolRuntime,
    *,
    replace_all: Annotated[
        bool,
        "If True, replace all occurrences of old_string. If False (default), old_string must be unique.",
    ] = False,
):
    """编辑文件内容"""
    resolved_backend = OpenSandbox(
        volumes=[
            Volume(
                name="workspace-root",
                host=Host(
                    path="/home/dev/liuyu/project/langchain-api/.langchain_api/workspace"
                ),
                mount_path="/workspace",
            )
        ]
    )
    try:
        validated_path = validate_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    res: EditResult = resolved_backend.edit(
        validated_path, old_string, new_string, replace_all=replace_all
    )
    resolved_backend.sandbox.kill()
    if res.error:
        return res.error
    if res.files_update is not None:
        return Command(
            update={
                "files": res.files_update,
                "messages": [
                    ToolMessage(
                        content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )
    return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"


GLOB_TIMEOUT = 20.0  # seconds


@tool("glob", description=GLOB_TOOL_DESCRIPTION)
def glob_tool(
    pattern: str,
    path: str = "/",
):
    """使用glob模式匹配文件"""
    try:
        validated_path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    resolved_backend = OpenSandbox(
        volumes=[
            Volume(
                name="workspace-root",
                host=Host(
                    path="/home/dev/liuyu/project/langchain-api/.langchain_api/workspace"
                ),
                mount_path="/workspace",
            )
        ]
    )
    try:
        validated_path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            resolved_backend.glob_info, pattern, path=validated_path
        )
        try:
            infos = future.result(timeout=GLOB_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."
    paths = [fi.get("path", "") for fi in infos]
    result = truncate_if_too_long(paths)
    resolved_backend.sandbox.kill()
    return str(result)


@tool("grep", description=GLOB_TOOL_DESCRIPTION)
def grep_tool(
    pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
    runtime: ToolRuntime,
    path: Annotated[
        str | None, "Directory to search in. Defaults to current working directory."
    ] = None,
    glob: Annotated[
        str | None, "Glob pattern to filter which files to search (e.g., '*.py')."
    ] = None,
    output_mode: Annotated[
        Literal["files_with_matches", "content", "count"],
        "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
    ] = "files_with_matches",
):
    """使用grep模式搜索文件内容"""
    resolved_backend = OpenSandbox(
        volumes=[
            Volume(
                name="workspace-root",
                host=Host(
                    path="/home/dev/liuyu/project/langchain-api/.langchain_api/workspace"
                ),
                mount_path="/workspace",
            )
        ]
    )
    raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
    if isinstance(raw, str):
        return raw
    formatted = format_grep_matches(raw, output_mode)
    result = truncate_if_too_long(formatted)
    resolved_backend.sandbox.kill()
    return result


# ['ls', 'read_file', 'write_file', 'edit_file', 'glob', 'grep']

if __name__ == "__main__":
    value = ls_tool.invoke({"path": "/workspace"})
    print(value)
    # value = execute_tool("env")
    # print(value)
