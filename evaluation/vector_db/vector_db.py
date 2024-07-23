from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

COLLECTION_NAME: str = "document"


def make_chroma_client(text_chunks: list[Document], embedding_function: Embeddings) -> Chroma:
    return Chroma.from_documents(text_chunks, embedding_function)