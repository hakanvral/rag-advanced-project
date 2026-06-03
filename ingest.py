import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore ,RetrievalMode

# database.py dosyasından gerekli ayarları çekiyoruz
from database import get_dense_embeddings, QDRANT_URL, COLLECTION_NAME ,get_sparse_embeddings


def ingest_single_pdf(file_path: str):

    print(f"{file_path} okunuyor ve Qdrant'a yükleniyor...")
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    if not documents:
        return "PDF içinden metin çikarilamadi."
        
    # SemanticChunker ->
    text_splitter = SemanticChunker(
        get_dense_embeddings(),
        breakpoint_threshold_type="percentile" # Anlamın en çok koptuğu %X'lik dilimlerde kes
    )

    chunks = text_splitter.split_documents(documents)
    
    # Qdrant'a yükleme (Metin vektörleri)
    QdrantVectorStore.from_documents(
        chunks,
        embedding=get_dense_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True,  
    )
    
    return f"Başarılı! {len(chunks)} parça akademik metin veri tabanına eklendi."
