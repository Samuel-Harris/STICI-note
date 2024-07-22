import chromadb

COLLECTION_NAME: str = "document"


def make_chroma_client(text_chunks: list[str]) -> chromadb.ClientAPI:
    chroma_client: chromadb.Client = chromadb.Client()
    collection: chromadb.Collection = chroma_client.create_collection(COLLECTION_NAME)
    collection.add(documents=text_chunks, ids=list(map(str, range(len(text_chunks)))))

    return chroma_client
