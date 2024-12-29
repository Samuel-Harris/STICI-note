from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
import random


def construct_chroma_vector_store(embedding_function: Embeddings) -> Chroma:
    return Chroma(collection_name=str(random.random()), embedding_function=embedding_function)
