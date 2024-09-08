from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


def construct_chroma_vector_store_retriever(embedding_function: Embeddings) -> VectorStoreRetriever:
    return Chroma(embedding_function=embedding_function).as_retriever()
