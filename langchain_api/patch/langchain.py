import itertools
from typing import Literal

from langchain_core.messages.ai import (
    LC_AUTO_PREFIX,
    LC_ID_PREFIX,
    AIMessageChunk,
    add_usage,
    create_tool_call_chunk,
    merge_content,
    merge_dicts,
    merge_lists,
)


def _add_ai_message_chunks(
    left: AIMessageChunk, *others: AIMessageChunk
) -> AIMessageChunk:
    """Add multiple `AIMessageChunk`s together.

    Args:
        left: The first `AIMessageChunk`.
        *others: Other `AIMessageChunk`s to add.

    Returns:
        The resulting `AIMessageChunk`.

    """
    content = merge_content(left.content, *(o.content for o in others))
    additional_kwargs = merge_dicts(
        left.additional_kwargs, *(o.additional_kwargs for o in others)
    )
    response_metadata = merge_dicts(
        left.response_metadata, *(o.response_metadata for o in others)
    )

    # Merge tool call chunks
    if raw_tool_calls := merge_lists(
        left.tool_call_chunks, *(o.tool_call_chunks for o in others)
    ):
        tool_call_chunks = [
            create_tool_call_chunk(
                name=rtc.get("name"),
                args=rtc.get("args"),
                index=rtc.get("index"),
                id=rtc.get("id"),
            )
            for rtc in raw_tool_calls
        ]
    else:
        tool_call_chunks = []

    # Token usage
    # Detect whether usage_metadata values represent cumulative streaming
    # counts (e.g. OpenAI with stream_usage=True sends running totals in
    # every chunk) vs. independent invocations that should be summed.
    # See: https://github.com/langchain-ai/langchain/issues/31351
    usages = [m.usage_metadata for m in (left, *others) if m.usage_metadata is not None]

    if not usages:
        usage_metadata = None
    else:
        inputs = [u.get("input_tokens", 0) for u in usages]
        totals = [u.get("total_tokens", 0) for u in usages]

        # Detect cumulative pattern:
        # constant input_tokens and monotonic increasing totals
        if len(set(inputs)) == 1 and totals == sorted(totals) and len(set(totals)) > 1:
            usage_metadata = usages[-1].copy()
        else:
            usage_metadata = usages[0].copy()
            for u in usages[1:]:
                usage_metadata = add_usage(usage_metadata, u)

    # Ranks are defined by the order of preference. Higher is better:
    # 2. Provider-assigned IDs (non lc_* and non lc_run-*)
    # 1. lc_run-* IDs
    # 0. lc_* and other remaining IDs
    best_rank = -1
    chunk_id = None
    candidates = itertools.chain([left.id], (o.id for o in others))

    for id_ in candidates:
        if not id_:
            continue

        if not id_.startswith(LC_ID_PREFIX) and not id_.startswith(LC_AUTO_PREFIX):
            chunk_id = id_
            # Highest rank, return instantly
            break

        rank = 1 if id_.startswith(LC_ID_PREFIX) else 0

        if rank > best_rank:
            best_rank = rank
            chunk_id = id_

    chunk_position: Literal["last"] | None = (
        "last" if any(x.chunk_position == "last" for x in [left, *others]) else None
    )

    return left.__class__(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_call_chunks=tool_call_chunks,
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        id=chunk_id,
        chunk_position=chunk_position,
    )


def patch_langchain():
    """用于修改langchain 的bug"""
    from langchain_core.messages import ai

    ai.add_ai_message_chunks = _add_ai_message_chunks
