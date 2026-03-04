from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Settings for the Bluesky Manager application, loaded from environment variables.
    """

    OPENAI_API_KEY: str
    HF_TOKEN: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()  # type: ignore
