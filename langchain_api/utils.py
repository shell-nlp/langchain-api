from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

from langchain_api.settings import settings

shanghai_tz = ZoneInfo("Asia/Shanghai")  # 设置亚洲/上海时区


def get_current_time() -> str:
    # 星期几的映射表
    weekday_map = {
        0: "星期一",
        1: "星期二",
        2: "星期三",
        3: "星期四",
        4: "星期五",
        5: "星期六",
        6: "星期日",
    }
    current_time = datetime.now(shanghai_tz)
    # 获取星期几（0=星期一，6=星期日）
    weekday_num = current_time.weekday()
    weekday_str = weekday_map[weekday_num]
    cur_time = f"""\n当前时间：{current_time.year}年{current_time.month}月{current_time.day}日 星期{weekday_str}"""
    return cur_time


def get_chat_model() -> ChatDeepSeek:

    return ChatDeepSeek(
        model=settings.CHAT_MODEL_NAME,
        api_base=settings.OPENAI_API_BASE,
        api_key=settings.OPENAI_API_KEY,
        extra_body={"enable_thinking": False},
    )


def get_embedding_model() -> OpenAIEmbeddings:

    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
    )
