from abc import abstractmethod, ABC

from langchain_text_splitters import TextSplitter
from pydantic.v1 import BaseModel


class TextSplitterWrapper(ABC, BaseModel):
    @abstractmethod
    def construct_text_splitter(self) -> TextSplitter:
        pass
