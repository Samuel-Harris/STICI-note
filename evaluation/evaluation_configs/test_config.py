from typing import Type

from langchain_core.prompts import PromptTemplate
from overrides import overrides

from evaluation.evaluation_configs.base_config import BaseConfig
from evaluation.evaluation_configs.model_config import ModelConfig
from evaluation.pipelines.pipeline import Pipeline
from evaluation.text_splitters.text_splitter_factory import TextSplitterWrapper


class TestConfig(BaseConfig):
    llm_config: ModelConfig
    text_splitter_wrapper: TextSplitterWrapper
    prompt_template: PromptTemplate
    pipeline_class: Type[Pipeline]
    embedding_model_name: str

    class Config:
        arbitrary_types_allowed = True

    @overrides
    def get_attributes(self) -> dict[str, str | int | float | bool]:
        attributes: dict[str, str | int | float | bool] = {**self.llm_config.get_attributes(),
                                                           **self.text_splitter_wrapper.get_attributes(),
                                                           "prompt_template": str(self.prompt_template),
                                                           "pipeline_class": self.pipeline_class.__name__,
                                                           "embedding_model_name": self.embedding_model_name}

        return attributes
