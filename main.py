from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from database import get_vector_store
from ingest import ingest_directory_documents

app = FastAPI(
    title="Local Academic RAG API",
    description="FastAPI, Qdrant ve Ollama (Llama3) tabanlı yerel akademik asistan API'si"
)

# Pydantic şeması (Gelen istek gövdesi için)
class QueryRequest(BaseModel):
    question: str

def format_docs(docs):
    """Gelen dökümanları birleştirir ve hangi PDF'ten geldiğini (metadata) belirtir."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Bilinmeyen Kaynak")
        page = doc.metadata.get("page", 0) + 1
        formatted.append(f"[Kaynak: {source}, Sayfa: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

@app.post("/ingest", summary="Data klasöründeki PDF'leri veri tabanına yükler")
def run_ingest():
    try:
        result = ingest_directory_documents()
        return {"status": "success", "message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest hatası: {str(e)}")

@app.post("/query", summary="Dokümanlara akademik soru sorar")
def answer_question(payload: QueryRequest):
    try:
        vector_store = get_vector_store()
        # En yakın 4 dökümanı getir
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Local Llama 3 bağlantısı
        llm = Ollama(model="llama3")
        
        template = """Sen gelişmiş bir akademik araştırmacısın. Aşağıdaki bağlam (context) bilgilerini 
        kullanarak sorulan soruyu bilimsel ve tarafsız bir dille cevapla. 
        Cevabını oluştururken bağlamda geçen kaynak bilgilerine sadık kal.
        Eğer bilgi bağlamda yoksa, 'Verilen dökümanlarda bu konuya dair bilgi bulunamadı.' de.
        
        Bağlam:
        {context}
        
        Soru: {question}
        
        Cevap:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # LCEL Zinciri
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Soruyu işlet ve cevabı dön
        answer = rag_chain.invoke(payload.question)
        return {"question": payload.question, "answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorgu işlenirken hata oluştu: {str(e)}")


from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
def root():
    # Doğrudan arayüze yönlendirir
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    # Uygulamayı 8000 portunda ayağa kaldırır
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)