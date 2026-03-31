import asyncio
import logging
import warnings

from deepagents.middleware.summarization import (
    Command,
    ContextOverflowError,
    ExtendedModelResponse,
    SummarizationEvent,
    SummarizationMiddleware,
    compute_summarization_defaults,
)
from langchain_api.middleware.sandbox.sandbox_system_tool import get_backend

logger = logging.getLogger(__name__)


def create_summarization_middleware(
    model,
    backend,
):
    from langchain.chat_models import (
        BaseChatModel as RuntimeBaseChatModel,
    )  # noqa: PLC0415

    if not isinstance(model, RuntimeBaseChatModel):
        msg = "`create_summarization_middleware` expects `model` to be a `BaseChatModel` instance."
        raise TypeError(msg)

    defaults = compute_summarization_defaults(model)
    return LangClawSummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=defaults["trigger"],
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
        truncate_args_settings=defaults["truncate_args_settings"],
    )


class LangClawSummarizationMiddleware(SummarizationMiddleware):
    def _get_backend(self, state, runtime):
        backend = get_backend(runtime, state)
        return backend

    def wrap_model_call(
        self,
        request,
        handler,
    ):
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Step 1: Truncate args if configured
        truncated_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        # Step 2: Check if summarization should happen
        counted_messages = (
            [request.system_message, *truncated_messages]
            if request.system_message is not None
            else truncated_messages
        )
        try:
            total_tokens = self.token_counter(
                counted_messages, tools=request.tools
            )  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        if not should_summarize:
            try:
                return handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                pass
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(
            truncated_messages, cutoff_index
        )

        # Offload to backend first so history is preserved before summarization.
        # If offload fails, summarization still proceeds (with file_path=None).
        backend = self._get_backend(request.state, request.runtime)
        file_path = self._offload_to_backend(backend, messages_to_summarize)
        if file_path is None:
            msg = "Offloading conversation history to backend failed during summarization. Older messages will not be recoverable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)

        # Generate summary
        summary = self._create_summary(messages_to_summarize)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[
                0
            ],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = handler(request.override(messages=modified_messages))
        # backend.sandbox.kill()
        # Return ExtendedModelResponse with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )

    async def awrap_model_call(
        self,
        request,
        handler,
    ):
        # Get effective messages based on previous summarization events
        effective_messages = self._get_effective_messages(request)

        # Step 1: Truncate args if configured
        truncated_messages, _ = self._truncate_args(
            effective_messages,
            request.system_message,
            request.tools,
        )

        # Step 2: Check if summarization should happen
        counted_messages = (
            [request.system_message, *truncated_messages]
            if request.system_message is not None
            else truncated_messages
        )
        try:
            total_tokens = self.token_counter(
                counted_messages, tools=request.tools
            )  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = self.token_counter(counted_messages)
        should_summarize = self._should_summarize(truncated_messages, total_tokens)

        # If no summarization needed, return with truncated messages
        if not should_summarize:
            try:
                return await handler(request.override(messages=truncated_messages))
            except ContextOverflowError:
                pass
                # Fallback to summarization on context overflow

        # Step 3: Perform summarization
        cutoff_index = self._determine_cutoff_index(truncated_messages)
        if cutoff_index <= 0:
            # Can't summarize, return truncated messages
            return await handler(request.override(messages=truncated_messages))

        messages_to_summarize, preserved_messages = self._partition_messages(
            truncated_messages, cutoff_index
        )

        # Offload to backend and generate summary concurrently -- they are independent.
        # If offload fails, summarization still proceeds (with file_path=None).
        backend = self._get_backend(request.state, request.runtime)
        file_path, summary = await asyncio.gather(
            self._aoffload_to_backend(backend, messages_to_summarize),
            self._acreate_summary(messages_to_summarize),
        )
        if file_path is None:
            msg = "Offloading conversation history to backend failed during summarization. Older messages will not be recoverable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)

        # Build summary message with file path reference
        new_messages = self._build_new_messages_with_path(summary, file_path)

        previous_event = request.state.get("_summarization_event")
        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)

        # Create new summarization event
        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[
                0
            ],  # The HumanMessage with summary  # ty: ignore[invalid-argument-type]
            "file_path": file_path,
        }

        # Modify request to use summarized messages
        modified_messages = [*new_messages, *preserved_messages]
        response = await handler(request.override(messages=modified_messages))
        backend.sandbox.kill()
        # Return ExtendedModelResponse with state update
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )
