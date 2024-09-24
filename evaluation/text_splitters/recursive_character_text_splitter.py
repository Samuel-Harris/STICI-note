from langchain.text_splitter import RecursiveCharacterTextSplitter

from evaluation.text_splitters.text_splitter_factory import TextSplitterWrapper


class RecursiveCharacterTextSplitterWrapper(TextSplitterWrapper):
    chunk_size: int
    chunk_overlap: int

    def construct_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )


RECURSIVE_CHARACTER_TEXT_SPLITTER_WRAPPER_LIST: list[RecursiveCharacterTextSplitterWrapper] = [
    RecursiveCharacterTextSplitterWrapper(chunk_size=300, chunk_overlap=100)]
