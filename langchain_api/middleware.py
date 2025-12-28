from typing import List, TypedDict
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents.middleware import (
    AgentMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel


class PlanningMiddleware(AgentMiddleware):
    """用于在代理中实现规划功能的中间件。每次通过总结上下文信息来规划下一步行动。并将规划结果添加到系统提示中。"""

    def __init__(self, model: str | BaseChatModel | None = None):
        from langchain_core.prompts import ChatPromptTemplate

        class Output(TypedDict):
            深度思考: str
            已完成的计划: List[str]
            下一步计划: List[str]

        structured_model = model.with_structured_output(schema=Output)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """# 你是一个规划专家。请总结当前的上下文信息，并规划下一步行动应该做什么？将你的规划结果输出为json格式,形如：
{{"深度思考": "...", "下一步计划": "..."}}

## 要求：
- 禁止私自在计划中得出任何上下文信息的结论，只能基于已有的上下文信息进行规划。

下面是对话历史：
""",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ]
        )
        self.chain = prompt | structured_model

    def wrap_model_call(self, request, handler):
        output = self.chain.invoke(
            {"chat_history": request.messages, "input": "规划："}
        )
        system_prompt = "根据规划结果来进行下一步行动。\n" + str(output)
        msgs = []
        for msg in request.messages:
            if "planning_middleware" in msg.additional_kwargs:
                continue
            msgs.append(msg)
        msgs.append(
            HumanMessage(
                content=system_prompt, additional_kwargs={"planning_middleware": True}
            )
        )
        return handler(request.override(messages=msgs))
