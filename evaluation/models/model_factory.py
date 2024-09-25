from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.language_models import LLM

from evaluation.evaluation_configs.model_config import ModelConfig


def construct_hf_model(llm_config: ModelConfig) -> LLM:
    return LlamaCpp(verbose=False, **llm_config.dict())