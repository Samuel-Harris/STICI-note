from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.language_models import LLM

from evaluation.evaluation_suite.model_config import ModelConfig


def construct_hf_model(model_config: ModelConfig) -> LLM:
    return LlamaCpp(**model_config.model_dump())