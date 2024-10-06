from typing import Optional

from overrides import overrides

from evaluation.evaluation_configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
    model_path: Optional[str] = "models/model_weights/tinyllama-1.1b-chat-v1.0.Q6_K.gguf"
    temperature: Optional[float] = 0.75
    max_tokens: Optional[int] = 2000
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 40
    last_n_tokens_size: Optional[int] = 64
    n_ctx: Optional[int] = 2048
    n_batch: Optional[int] = 512
    n_gpu_layers: Optional[int] = -1
    f16_kv: Optional[bool] = True

    @overrides
    def get_attributes(self) -> dict[str, str | int | float | bool]:
        return self.model_dump()
