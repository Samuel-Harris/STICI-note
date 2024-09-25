from overrides import overrides

from evaluation.evaluation_configs.base_config import BaseConfig


class ModelConfig(BaseConfig):
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

    @overrides
    def get_attributes(self) -> dict[str, str|int|float|bool]:
        return self.dict()
