from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_huggingface import HuggingFaceEmbeddings

from evaluation.evaluation_configs.test_config import TestConfig
from evaluation.models.model_factory import construct_hf_model
from evaluation.pipelines.pipeline import Pipeline
from evaluation.vector_db.vector_db import construct_chroma_vector_store


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
