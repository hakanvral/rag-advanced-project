from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings  # GÜNCELLENDİ

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "academic_papers"
EMBEDDING_MODEL = "nomic-embed-text"

def get_embeddings():
    """Ollama embedding modelini döner."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)

def get_vector_store():
    """Qdrant vektör mağazasına bağlanır ve nesneyi döner."""
    client = QdrantClient(url=QDRANT_URL)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )
    return vector_store