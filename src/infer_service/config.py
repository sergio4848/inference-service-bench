"""Runtime configuration through INFER_* environment variables or a .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFER_", env_file=".env", extra="ignore")

    # Vision model (ONNX). See scripts/get_model.py for the reference MobileNetV2 download.
    model_path: str = "models/mobilenetv2-12.onnx"
    labels_path: str = "models/synset.txt"
    input_size: int = 224
    intra_op_threads: int = 0  # 0 = ONNX Runtime default
    # Micro-batching: collect up to `batch_max` requests or wait `batch_wait_ms`, whichever comes first.
    # Default 1 = off. Measured on CPU it adds queue wait without throughput (see README); switch it
    # on for accelerators where one forward pass has a large fixed cost.
    batch_max: int = 1
    batch_wait_ms: float = 4.0
    top_k: int = 5

    # LLM gateway: any OpenAI-compatible chat-completions endpoint.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None  # falls back to OPENAI_API_KEY
    llm_model: str = "gpt-4.1-mini"
    llm_timeout_s: float = 60.0

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
