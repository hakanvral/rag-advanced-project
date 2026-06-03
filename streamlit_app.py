import streamlit as st
import requests
import time

# FastAPI Sunucu Adresi
BASE_URL = "http://127.0.0.1:8000"

# 1. Düzen: Sohbet deneyimi için "centered" (ortalanmış) her zaman daha şıktır.
st.set_page_config(page_title="Akademik Asistan", page_icon="🪶", layout="centered")

# 2. Minimalist CSS Enjeksiyonu
st.markdown("""
    <style>
    /* Varsayılan Streamlit menülerini ve footer'ı gizle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Butonları daha yumuşak ve şık yap */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        color: #333333;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #9e9e9e;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #000000;
    }
    
    /* Chat giriş kutusunun alt boşluğunu ayarla */
    .stChatFloatingInputContainer {
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Şık ve Sade Ana Başlık
st.markdown("<h2 style='text-align: center; color: #2e3b4e; margin-bottom: 0;'>Akademik Araştırma Asistanı</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 14px; margin-bottom: 2rem;'>Yapay Zeka Destekli Belge Analizi</p>", unsafe_allow_html=True)

# --- SideBar (Minimalist) ---
with st.sidebar:
    st.markdown("### 📄 Belge Yönetimi")
    
    # Etiketi gizlenmiş daha temiz bir yükleme alanı
    uploaded_file = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Belgeyi İşle"):
            with st.spinner("Vektör uzayına aktarılıyor..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        # Ekranda yer kaplamayan, zarif pop-up bildirimi (Toast)
                        st.toast(f"'{uploaded_file.name}' başarıyla işlendi!", icon="✅")
                        time.sleep(1) 
                    else:
                        st.error(f"Hata: {response.json().get('detail')}")
                except Exception as e:
                    st.error("Sunucuya ulaşılamadı. FastAPI açık mı?")

    st.markdown("---")
    st.markdown("### ⚙️ Sistem Durumu")
    # Kalabalık arka plan renkleri yerine zarif metin göstergeleri
    st.markdown("🟢 **LLM:** Aktif")
    st.markdown("🟢 **Vektör DB:** Aktif")

# --- Sohbet Arayüzü ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Giriş kutusundaki metni de daha nazik bir tona çektik
if prompt := st.chat_input("Belgeyle ilgili ne öğrenmek istersiniz?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"question": prompt},
                stream=True
            )
            
            if response.status_code == 200:
                def stream_generator():
                    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                        if chunk:
                            yield chunk
                            
                full_answer = st.write_stream(stream_generator())
                st.session_state.messages.append({"role": "assistant", "content": full_answer})
            else:
                try:
                    error_msg = response.json().get("detail", "Hata oluştu.")
                except:
                    error_msg = response.text
                st.error(f"Sistem Hatası: {error_msg}")
                
        except Exception as e:
            st.error("Bağlantı Hatası: Lütfen backend sunucusunu kontrol edin.")