from abc import abstractmethod

from langchain_text_splitters import TextSplitter

from evaluation.evaluation_configs.base_config import BaseConfig


class TextSplitterWrapper(BaseConfig):
    @abstractmethod
    def construct_text_splitter(self) -> TextSplitter:
        pass
