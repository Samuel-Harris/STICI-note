import json
from itertools import product
from typing import Type, Generator

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TextSplitter
from pydantic.v1 import BaseModel
import pandas as pd

from evaluation.models.model_factory import construct_hf_model
from evaluation.pipelines.basic_pipeline import BasicPipeline
from evaluation.pipelines.pipeline import Pipeline
from evaluation.evaluation_configs.model_config import ModelConfig
from evaluation.text_splitters.recursive_character_text_splitter import RECURSIVE_CHARACTER_TEXT_SPLITTER_WRAPPER_LIST
from evaluation.text_splitters.text_splitter_factory import TextSplitterWrapper
from evaluation.vector_db.vector_db import construct_chroma_vector_store

# import and set up config data
with open("evaluation_configs/model_configs.json", "r") as f:
    raw_model_configs = json.loads(f.read())
    model_configs = [ModelConfig.validate(raw_model_config) for raw_model_config in raw_model_configs]

with open("evaluation_configs/prompt_templates.json", "r") as f:
    raw_prompt_templates = json.loads(f.read())
    prompt_templates = [PromptTemplate(template=raw_prompt_template, input_variables=["input", "context"]) for
                        raw_prompt_template in raw_prompt_templates]

with open("evaluation_configs/embedding_models.json", "r") as f:
    embedding_models = json.loads(f.read())

pipeline_classes: list[Type[Pipeline]] = [
    BasicPipeline
]

text_splitter_wrappers: list[TextSplitter] = [
    *RECURSIVE_CHARACTER_TEXT_SPLITTER_WRAPPER_LIST,
]


class TestConfig(BaseModel):
    llm_config: ModelConfig
    text_splitter_wrapper: TextSplitterWrapper
    prompt_template: PromptTemplate
    pipeline_class: Type[Pipeline]
    embedding_model_name: str

    class Config:
        arbitrary_types_allowed = True


class TestPipeline:
    def __init__(self, test_config: TestConfig) -> None:
        self.test_config: TestConfig = test_config

        model: LLM = construct_hf_model(test_config.llm_config)
        embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name=test_config.embedding_model_name)
        self.vector_store: Chroma = construct_chroma_vector_store(embedding_model)

        self.pipeline: Pipeline = test_config.pipeline_class(model, test_config.prompt_template,
                                                             self.vector_store.as_retriever())

        self.text_splitter = self.test_config.text_splitter_wrapper.construct_text_splitter()

    def add_text_to_vector_db(self, text: str) -> None:
        chunks: list[str] = self.text_splitter.split_text(text)
        documents = [Document(chunk) for chunk in chunks]
        self.vector_store.add_documents(documents)

    def reset_vector_db(self) -> None:
        self.vector_store.reset_collection()


class TestPipelineContextManager:
    def __init__(self, test_pipeline: TestPipeline, row: pd.Series, documents_df: pd.DataFrame) -> None:
        self.test_pipeline: TestPipeline = test_pipeline
        self.document_text: str = documents_df[documents_df["index"] == row["document_index"]]["text"].iloc[0]

    def __enter__(self) -> TestPipeline:
        self.test_pipeline.add_text_to_vector_db(self.document_text)

        return self.test_pipeline

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.test_pipeline.reset_vector_db()


def generate_test_pipeline_df() -> pd.DataFrame:
    test_pipeline_list: list[TestPipeline] = []

    model_config: ModelConfig
    pipeline_class: Type[Pipeline]
    prompt_template: PromptTemplate
    embedding_model_name: str
    for model_config, pipeline_class, text_splitter_wrapper, prompt_template, embedding_model_name in product(
            model_configs,
            pipeline_classes,
            text_splitter_wrappers,
            prompt_templates,
            embedding_models):
        test_config: TestConfig = TestConfig(llm_config=model_config, prompt_template=prompt_template,
                                             text_splitter_wrapper=text_splitter_wrapper, pipeline_class=pipeline_class,
                                             embedding_model_name=embedding_model_name)
        test_pipeline: TestPipeline = TestPipeline(test_config)
        test_pipeline_list.append(test_pipeline)

    return pd.DataFrame(test_pipeline_list)
