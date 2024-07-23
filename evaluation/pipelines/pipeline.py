from abc import ABC, abstractmethod

import chromadb
from langchain.prompts import PromptTemplate


class Pipeline(ABC):
    @abstractmethod
    def __init__(self, temperature: float, prompt_template: PromptTemplate, vector_db: chromadb.ClientAPI) -> None:
        pass

    @abstractmethod
    def query(self, query: str) -> str:
        pass
