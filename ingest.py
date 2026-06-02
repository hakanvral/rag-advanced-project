import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

# database.py dosyasından gerekli ayarları çekiyoruz
from database import get_embeddings, QDRANT_URL, COLLECTION_NAME

DATA_DIR = "./Data"

def ingest_directory_documents():
    """Data klasöründeki tüm PDF'leri okur ve Qdrant'a yükler."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return "Data klasörü bulunamadı, yeni bir tane oluşturuldu. Lütfen içine PDF atın."

    print(f"'{DATA_DIR}' klasöründeki dökümanlar taranıyor...")
    # Klasör içindeki tüm PDF'leri otomatik algılar
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        return "Klasörün içinde işlenecek PDF bulunamadı."
        
    print(f"Toplam {len(documents)} sayfa döküman bulundu. Parçalanıyor (Chunking)...")
    
    # Akademik metinler için optimize edilmiş chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"{len(chunks)} adet metin parçası Qdrant'a yükleniyor...")
    
    # from_documents metodu: Koleksiyon yoksa sıfırdan oluşturur, varsa üstüne ekler
    QdrantVectorStore.from_documents(
        chunks,
        get_embeddings(),
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    
    return f"Başarılı! {len(chunks)} parça veri Qdrant'a indekslendi."

if __name__ == "__main__":
    # Test etmek için direkt çalıştırılabilir
    print(ingest_directory_documents())