from abc import ABC, abstractmethod

from langchain.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelLike
from langchain_core.vectorstores import VectorStoreRetriever


class Pipeline(ABC):
    @abstractmethod
    def __init__(self, llm: LanguageModelLike, prompt_template: PromptTemplate,
                 vector_db_retriever: VectorStoreRetriever) -> None:
        pass

    @abstractmethod
    def query(self, query: str) -> str:
        pass
