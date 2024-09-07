import json
from itertools import product
from typing import Type

from langchain_core.language_models import LLM, LanguageModelLike
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic.v1 import BaseModel

from evaluation.models.model_factory import construct_hf_model
from evaluation.pipelines.basic_pipeline import BasicPipeline
from evaluation.pipelines.pipeline import Pipeline
from evaluation.evaluation_configs.model_config import ModelConfig

# import and set up config data
with open("evaluation/evaluation_configs/model_configs.json", "r") as f:
    raw_model_configs = json.loads(f.read())
    model_configs = [ModelConfig.validate(raw_model_config) for raw_model_config in raw_model_configs]

with open("evaluation/evaluation_configs/prompt_templates.json", "r") as f:
    raw_prompt_templates = json.loads(f.read())
    prompt_templates = [PromptTemplate(template=raw_prompt_template, input_variables=["input", "context"]) for
                        raw_prompt_template in raw_prompt_templates]

with open("evaluation/evaluation_configs/embedding_models.json", "r") as f:
    embedding_models = json.loads(f.read())

pipeline_classes: list[Type[Pipeline]] = [
    BasicPipeline
]


class TestConfig(BaseModel):
    llm_config: ModelConfig
    model: LanguageModelLike
    prompt_template: PromptTemplate
    pipeline_class: Type[Pipeline]
    embedding_model: HuggingFaceEmbeddings

    class Config:
        arbitrary_types_allowed = True


class TestConfigIterator:
    def __init__(self):
        self.current = 0
        self.test_configs: list[TestConfig] = []

        model_config: ModelConfig
        pipeline_class: Type[Pipeline]
        prompt_template: PromptTemplate
        embedding_model_name: str
        for model_config, pipeline_class, prompt_template, embedding_model_name in product(model_configs,
                                                                                           pipeline_classes,
                                                                                           prompt_templates,
                                                                                           embedding_models):
            model: LLM = construct_hf_model(model_config)
            embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            test_config = TestConfig(llm_config=model_config, model=model, prompt_template=prompt_template,
                                     pipeline_class=pipeline_class, embedding_model=embedding_model)
            self.test_configs.append(test_config)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < len(self.test_configs):
            test_config = self.test_configs[self.current]
            self.current += 1

            return test_config
        else:
            raise StopIteration


for x in TestConfigIterator():
    print(x)