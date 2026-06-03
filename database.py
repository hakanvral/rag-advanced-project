from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore , RetrievalMode ,FastEmbedSparse
from langchain_ollama.embeddings import OllamaEmbeddings  # GÜNCELLENDİ

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
EMBEDD_MODEL = "bge-m3:latest"


def get_dense_embeddings():
    return OllamaEmbeddings(model=EMBEDD_MODEL)


def get_sparse_embeddings():
    return FastEmbedSparse(model_name="Qdrant/bm25")

def get_vector_store():
    
    client = QdrantClient(url=QDRANT_URL)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
    )
    return vector_store