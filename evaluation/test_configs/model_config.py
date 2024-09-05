from pydantic.v1 import BaseModel


class ModelConfig(BaseModel):
    model_path: str
    temperature: float
    max_tokens: int
    top_p: int
    top_k: int
    last_n_tokens_size: int
    n_ctx: int
    n_batch: int
    n_gpu_layers: int
    f16_kv: bool
