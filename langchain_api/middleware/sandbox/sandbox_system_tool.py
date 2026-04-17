import asyncio
import base64
import concurrent.futures
from pathlib import Path
from typing import Annotated, Literal, NotRequired, cast

from deepagents.backends.utils import (
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
    validate_path,
)
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import (
    DEFAULT_READ_LIMIT,
    DEFAULT_READ_OFFSET,
    EDIT_FILE_TOOL_DESCRIPTION,
    EXECUTE_TOOL_DESCRIPTION,
    GLOB_TOOL_DESCRIPTION,
    IMAGE_EXTENSIONS,
    IMAGE_MEDIA_TYPES,
    LIST_FILES_TOOL_DESCRIPTION,
    NUM_CHARS_PER_TOKEN,
    READ_FILE_TOOL_DESCRIPTION,
    READ_FILE_TRUNCATION_MSG,
    WRITE_FILE_TOOL_DESCRIPTION,
    EditResult,
    WriteResult,
    _build_evicted_content,
    _create_content_preview,
    _extract_text_from_message,
    create_image_block,
)
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.types import Command
from loguru import logger
from opensandbox.models.sandboxes import Host, Volume

from langchain_api.agent.context import AgentContext as LangclawContext
from langchain_api.backend.open_sandbox import OpenSandbox

FILESYSTEM_SYSTEM_PROMPT = """## Following Conventions

- Read files before editing — understand existing content before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /. Follow the tool docs for the available tools, and use pagination (offset/limit) when reading large files.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files

## Large Tool Results

When a tool result is too large, it may be offloaded into the filesystem instead of being returned inline. In those cases, use `read_file` to inspect the saved result in chunks, or use `grep` within `/large_tool_results/` if you need to search across offloaded tool results and do not know the exact file path. Offloaded tool results are stored under `/large_tool_results/<tool_call_id>`."""
EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)
TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""


workspace_path = Path(__file__).parent.parent.parent.parent / "my_workspace"
token_limit = 20000
GLOB_TIMEOUT = 20.0  # seconds


def get_user_workspace_path(user_id: str) -> str:
    new_workspace_path = Path(f"{workspace_path}/{user_id}/.langclaw")
    new_workspace_path.mkdir(parents=True, exist_ok=True)
    conversation_history = new_workspace_path / "conversation_history"
    conversation_history.mkdir(parents=True, exist_ok=True)
    return str(new_workspace_path)


def get_existing_backend(
    sandbox_id: str,
) -> OpenSandbox:
    return OpenSandbox(
        existing_sandbox_id=sandbox_id,
    )


def get_new_backend(
    user_id: str,
) -> OpenSandbox:
    workspace_path = get_user_workspace_path(user_id)
    return OpenSandbox(
        volumes=[
            Volume(
                name=f"langclaw-{user_id}",
                host=Host(path=workspace_path),
                mount_path="/.langclaw",
            ),
            Volume(
                name=f"langclaw-conversation-history-{user_id}",
                host=Host(path=workspace_path + "/conversation_history"),
                mount_path="/conversation_history",
            ),
        ]
    )


def get_backend(
    runtime: ToolRuntime[LangclawContext, None] | Runtime[LangclawContext],
    state: dict | None = None,
) -> OpenSandbox:
    user_id = runtime.context.user_id

    # 优先使用传入的state，否则从runtime中获取
    target_state = (
        state
        if state is not None
        else (runtime.state if isinstance(runtime, ToolRuntime) else None)
    )

    if target_state:
        sandbox_id_dict: dict = target_state.get("sandbox_id_dict", {})
        if (
            user_id in sandbox_id_dict and sandbox_id_dict[user_id] is not None
        ):  # 说明已经创建好了沙箱，可以直接使用
            sandbox_id = sandbox_id_dict[user_id]
            logger.warning(f"得到用户 **{user_id}** 已存在的沙箱 ID: {sandbox_id}")
            return get_existing_backend(sandbox_id)

    # 未找到沙箱ID或没有状态，创建新沙箱
    backend = get_new_backend(user_id)
    logger.warning(f"创建用户 **{user_id}** 新沙箱, 沙箱 ID 为 {backend.sandbox.id}")
    return backend


@tool("ls", description=LIST_FILES_TOOL_DESCRIPTION)
def ls_tool(
    path: Annotated[
        str, "Absolute path to the directory to list. Must be absolute, not relative."
    ],
    runtime: ToolRuntime[LangclawContext, None],
):
    """列出当前目录下的文件"""
    backend = get_backend(runtime)
    try:
        validated_path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"

    infos = backend.ls_info(path=validated_path)
    paths = [fi.get("path", "") for fi in infos]
    result = truncate_if_too_long(paths)
    # backend.sandbox.kill()
    return str(result)


@tool("execute", description=EXECUTE_TOOL_DESCRIPTION)
def execute_tool(
    command: Annotated[str, "Shell command to execute in the sandbox environment."],
    runtime: ToolRuntime[LangclawContext, None],
    timeout: Annotated[
        int | None,
        "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
    ] = None,
):
    """执行命令并返回结果"""
    resolved_backend = get_backend(runtime)
    result = resolved_backend.execute(command)
    # Format output for LLM consumption
    parts = [result.output]

    # if result.exit_code is not None:
    #     status = "succeeded" if result.exit_code == 0 else "failed"
    #     parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

    if result.truncated:
        parts.append("\n[Output was truncated due to size limits]")
    # resolved_backend.sandbox.kill()
    return "".join(parts)


@tool("read_file", description=READ_FILE_TOOL_DESCRIPTION)
def read_file(
    file_path: Annotated[
        str, "Absolute path to the file to read. Must be absolute, not relative."
    ],
    runtime: ToolRuntime[LangclawContext, None],
    offset: Annotated[
        int,
        "Line number to start reading from (0-indexed). Use for pagination of large files.",
    ] = DEFAULT_READ_OFFSET,
    limit: Annotated[
        int, "Maximum number of lines to read. Use for pagination of large files."
    ] = DEFAULT_READ_LIMIT,
):
    """读取文件内容"""
    backend = get_backend(runtime)
    try:
        validated_path = validate_path(file_path)
    except ValueError as e:
        return f"Error: {e}"

    ext = Path(validated_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        responses = backend.download_files([validated_path])
        if responses and responses[0].content is not None:
            media_type = IMAGE_MEDIA_TYPES.get(ext, "image/png")
            image_b64 = base64.standard_b64encode(responses[0].content).decode("utf-8")
            # backend.sandbox.kill()
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
    # backend.sandbox.kill()
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
    runtime: ToolRuntime[LangclawContext, None],
):
    """写入文件内容"""
    backend = get_backend(runtime)
    try:
        validated_path = validate_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    res: WriteResult = backend.write(validated_path, content)
    if res.error:
        return res.error
    # If backend returns state update, wrap into Command with ToolMessage
    # backend.sandbox.kill()
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
    runtime: ToolRuntime[LangclawContext, None],
    *,
    replace_all: Annotated[
        bool,
        "If True, replace all occurrences of old_string. If False (default), old_string must be unique.",
    ] = False,
):
    """编辑文件内容"""
    resolved_backend = get_backend(runtime)
    try:
        validated_path = validate_path(file_path)
    except ValueError as e:
        return f"Error: {e}"
    res: EditResult = resolved_backend.edit(
        validated_path, old_string, new_string, replace_all=replace_all
    )
    # resolved_backend.sandbox.kill()
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


@tool("glob", description=GLOB_TOOL_DESCRIPTION)
def glob_tool(
    pattern: str,
    runtime: ToolRuntime[LangclawContext, None],
    path: str = "/",
):
    """使用glob模式匹配文件"""
    try:
        validated_path = validate_path(path)
    except ValueError as e:
        return f"Error: {e}"
    resolved_backend = get_backend(runtime)
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
    # resolved_backend.sandbox.kill()
    return str(result)


@tool("grep", description=GLOB_TOOL_DESCRIPTION)
def grep_tool(
    pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
    runtime: ToolRuntime[LangclawContext, None],
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

    resolved_backend = get_backend(runtime)
    raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
    if isinstance(raw, str):
        return raw
    formatted = format_grep_matches(raw, output_mode)
    result = truncate_if_too_long(formatted)
    # resolved_backend.sandbox.kill()
    return result


def dict_merge(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    if left is None:
        left = {}
    result = left | right
    return result


class SandboxSystemToolState(AgentState):
    """沙箱系统工具中间件状态"""

    sandbox_id_dict: NotRequired[Annotated[dict[str, str | None], dict_merge]]


class SandboxSystemToolMiddleware(AgentMiddleware):
    """沙箱系统工具中间件"""

    tools = [
        execute_tool,
        ls_tool,
        read_file,
        write_file,
        grep_tool,
        glob_tool,
        edit_file,
    ]
    state_schema = SandboxSystemToolState

    def __init__(self, tool_token_limit_before_evict: int | None = 20000) -> None:
        self._tool_token_limit_before_evict = tool_token_limit_before_evict

    def before_agent(self, state, runtime):
        user_id = runtime.context.user_id
        backend = get_backend(runtime, state)
        sandbox_id = backend.sandbox.id
        return {"sandbox_id_dict": {user_id: sandbox_id}}

    async def abefore_agent(self, state, runtime):
        return await asyncio.to_thread(self.before_agent, state, runtime)

    def after_agent(self, state, runtime):
        """目的：在代理执行完成后，统一杀死用户的沙箱环境"""

        user_id = runtime.context.user_id
        backend = get_backend(runtime, state)
        sandbox_id = backend.sandbox.id
        backend.sandbox.kill()
        logger.warning(f"用户 **{user_id}** 的沙箱 ID 为 {sandbox_id} 已被杀死")
        return {"sandbox_id_dict": {user_id: None}}

    async def aafter_agent(self, state, runtime):
        return await asyncio.to_thread(self.after_agent, state, runtime)

    def wrap_model_call(
        self,
        request,
        handler,
    ):
        prompt_parts = [FILESYSTEM_SYSTEM_PROMPT, EXECUTION_SYSTEM_PROMPT]
        system_prompt = "\n\n".join(prompt_parts).strip()
        new_system_message = append_to_system_message(
            request.system_message, system_prompt
        )
        request = request.override(system_message=new_system_message)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        return await self.wrap_model_call(request, handler)

    def wrap_tool_call(self, request, handler):
        if (
            self._tool_token_limit_before_evict is None
            or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION
        ):
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(self, request, handler):
        if (
            self._tool_token_limit_before_evict is None
            or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION
        ):
            return await handler(request)

        tool_result = await handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend,
    ):
        if not self._tool_token_limit_before_evict:
            return message, None

        content_str = _extract_text_from_message(message)

        # Check if content exceeds eviction threshold
        if (
            len(content_str)
            <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict
        ):
            return message, None

        # Write content to filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, None

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        evicted = _build_evicted_content(message, replacement_text)
        processed_message = ToolMessage(
            content=cast("str | list[str | dict]", evicted),
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=message.artifact,
            status=message.status,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
        )
        return processed_message, result.files_update

    def _intercept_large_tool_result(
        self, tool_result: ToolMessage | Command, runtime: ToolRuntime
    ) -> ToolMessage | Command:
        if isinstance(tool_result, ToolMessage):
            resolved_backend = get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(
                update={
                    **update,
                    "messages": processed_messages,
                    "files": accumulated_file_updates,
                }
            )
        msg = f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)
