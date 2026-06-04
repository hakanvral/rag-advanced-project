from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from database import get_vector_store

LLM_MODEL = "qwen2:7B"

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Bilinmeyen Kaynak")
        page = doc.metadata.get("page", 0) + 1
        formatted.append(f"[Kaynak: {source}, Sayfa: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

async def generate_rag_stream(question: str):
    """LangChain RAG zincirini çalıştırır ve cevabı parça parça akıtır."""
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = Ollama(model=LLM_MODEL) 
    
    template = """You are an elite academic research assistant. Your primary task is to answer the user's query comprehensively and objectively, using ONLY the information provided in the Context below.

    CRITICAL RULES:
    1. NO HALLUCINATION: If the answer cannot be deduced from the provided Context, you must explicitly state: "Verilen dökümanlarda bu konuya dair spesifik bir bilgi bulunmamaktadır." Do not use outside knowledge.
    2. ACADEMIC TONE: Maintain a formal, objective, and scholarly tone.
    3. LANGUAGE: You must ALWAYS generate your final response in fluent, flawless Turkish, regardless of the language of the prompt or context.
    4. CITATION (MANDATORY): You MUST cite the exact source document and page number for every major claim or fact you provide. Use the metadata provided in the context. Format your citations inline at the end of the sentence like this: (Kaynak: makale_adi.pdf, Sayfa: X).
    
    Context information is below:
    ---------------------
    {context}
    ---------------------
    
    User's Question: {question}
    
    Answer in Turkish:"""
    
    prompt = PromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    for chunk in rag_chain.stream(question):
        yield chunk
