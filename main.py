from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, RedirectResponse
import shutil
import os

# Yeni mimariye uygun importlarımız
from database import get_vector_store
from ingest import ingest_single_pdf
from schemas import QueryRequest
from rag_service import generate_rag_stream

app = FastAPI(
    title="Local Academic RAG API",
    description="FastAPI ve modüler servis mimarisine sahip Akademik Asistan"
)

@app.post("/upload", summary="Arayüzden gelen PDF'i Qdrant'a ekler")
async def upload_and_ingest(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"
    try:
        # Dosyayı Streamlit'ten al ve geçici olarak diske kaydet
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Dosyayı Qdrant'a işle
        result = ingest_single_pdf(temp_file_path)
        
        # İşlem bitince geçici PDF'i sil
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return {"status": "success", "message": result}
        
    except Exception as e:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Dosya işleme hatası: {str(e)}")

@app.post("/query", summary="Dokümanlara akademik soru sorar (Streaming)")
async def answer_question(payload: QueryRequest):
    try:
        # Tüm yapay zeka işlemleri rag_service üzerinden çağrılır
        return StreamingResponse(
            generate_rag_stream(payload.question), 
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorgu işlenirken hata oluştu: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    # Doğrudan arayüze yönlendirir
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    # Uygulamayı 8000 portunda ayağa kaldırır
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)