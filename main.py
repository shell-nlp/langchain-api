from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")


async def main():
    async for chunk in client.runs.stream(
        thread_id=None,
        assistant_id="agent",
        input={
            "messages": [
                {
                    "role": "human",
                    "content": "你好",
                }
            ],
        },
        stream_mode=["messages", "updates"],
    ):
        print(chunk)
        print("\n\n")


asyncio.run(main())
