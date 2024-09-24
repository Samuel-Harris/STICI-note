from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def construct_chroma_vector_store(embedding_function: Embeddings) -> Chroma:
    return Chroma(embedding_function=embedding_function)
