from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import find_dotenv, load_dotenv

# 获取.env文件路径
env_path = find_dotenv(filename=".env", raise_error_if_not_found=True)

# 再将.env文件内容加载到环境变量中
load_dotenv()


class Settings(BaseSettings):
    # openAI api_base 和 api_key配置
    OPENAI_API_BASE: str
    CHAT_MODEL_NAME: str

    # Chat模型配置
    OPENAI_API_KEY: str

    # Embedding模型配置
    EMBEDDING_MODEL_NAME: str

    # elasticsearch配置
    ES_URL: str
    ES_URSR: str
    ES_PWD: str

    model_config = SettingsConfigDict(
        env_file=str(env_path),
        env_file_encoding="utf-8",
    )


def get_settings() -> Settings:
    return Settings()


settings = get_settings()

if __name__ == "__main__":
    print(settings.CHAT_MODEL_NAME)
