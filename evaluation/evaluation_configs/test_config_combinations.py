import json
from itertools import product
from typing import Type, Generator

from langchain_core.prompts import PromptTemplate

from evaluation.evaluation_configs.model_config import ModelConfig
from evaluation.evaluation_configs.test_config import TestConfig
from evaluation.evaluation_configs.test_pipeline import TestPipeline
from evaluation.pipelines.basic_pipeline import BasicPipeline
from evaluation.pipelines.pipeline import Pipeline
from evaluation.text_splitters.recursive_character_text_splitter import RecursiveCharacterTextSplitterWrapper


class TestConfigCombinations:
    test_configs: list[TestConfig]
    model_configs: list[ModelConfig]
    prompt_templates: list[PromptTemplate]
    embedding_models: list[str]
    pipeline_classes: list[Type[Pipeline]] = [BasicPipeline]
    text_splitter_wrappers: list[RecursiveCharacterTextSplitterWrapper] = [
        RecursiveCharacterTextSplitterWrapper(chunk_size=300, chunk_overlap=100)]

    def __init__(self):
        # import and set up config data
        with open("evaluation_configs/model_configs.json", "r") as f:
            raw_model_configs: dict = json.loads(f.read())

            self.model_configs = [ModelConfig.model_validate(dict(zip(raw_model_configs.keys(), model_config))) for
                                  model_config
                                  in
                                  product(*raw_model_configs.values())]

        with open("evaluation_configs/prompt_templates.json", "r") as f:
            raw_prompt_templates = json.loads(f.read())
            self.prompt_templates = [PromptTemplate(template=raw_prompt_template, input_variables=["input", "context"])
                                     for
                                     raw_prompt_template in raw_prompt_templates]

        with open("evaluation_configs/embedding_models.json", "r") as f:
            self.embedding_models = json.loads(f.read())

        # Generate all possible config combinations
        self.test_configs = []

        model_config: ModelConfig
        pipeline_class: Type[Pipeline]
        prompt_template: PromptTemplate
        embedding_model_name: str
        for model_config, prompt_template, embedding_model_name, pipeline_class, text_splitter_wrapper in product(
                self.model_configs,
                self.prompt_templates,
                self.embedding_models,
                self.pipeline_classes,
                self.text_splitter_wrappers):
            self.test_configs.append(TestConfig(llm_config=model_config, prompt_template=prompt_template,
                                                text_splitter_wrapper=text_splitter_wrapper,
                                                pipeline_class=pipeline_class,
                                                embedding_model_name=embedding_model_name))

    def generate_test_pipelines(self) -> Generator[TestPipeline, None, None]:
        test_config: TestConfig
        for test_config in self.test_configs:
            yield TestPipeline(test_config)

    def get_test_config_combinations(self):
        return self.test_configs

    def get_n_combinations(self) -> int:
        return len(self.test_configs)
