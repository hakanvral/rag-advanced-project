import streamlit as st
import requests

# FastAPI Sunucu Adresi
BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Akademik RAG Asistanı", page_icon="🎓", layout="wide")

# CSS ile biraz özelleştirme
st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .sidebar .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Akademik RAG Araştırma Asistanı")
st.subheader("Qdrant & Llama 3 ile Güçlendirilmiş Yerel Analiz")

# --- SideBar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3408/3408455.png", width=100)
    st.header("Yönetim Paneli")
    
    st.info("'Data' klasörüne yeni PDF eklediyseniz aşağıdaki butona basın.")
    
    if st.button("📚 Veri Tabanını Güncelle (Ingest)"):
        with st.spinner("PDF'ler taranıyor ve Qdrant'a yükleniyor..."):
            try:
                response = requests.post(f"{BASE_URL}/ingest")
                if response.status_code == 200:
                    st.success("Veri tabanı başarıyla güncellendi!")
                else:
                    st.error(f"Hata: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Bağlantı Hatası: FastAPI sunucusu çalışıyor mu? \n{e}")

    st.divider()
    st.write("**Sistem Durumu:**")
    st.success("Ollama: Aktif")
    st.success("Qdrant: Aktif")

# --- Sohbet Arayüzü ---

# Mesaj geçmişini başlat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan soru al
if prompt := st.chat_input("Dökümanlar hakkında bir soru sorun (Örn: Bu çalışmanın metodolojisi nedir?)"):
    # Kullanıcı mesajını göster ve kaydet
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # FastAPI'ye sorgu gönder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Bağlam taranıyor ve cevap üretiliyor...*")
        
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"question": prompt}
            )
            
            if response.status_code == 200:
                answer = response.json().get("answer")
                message_placeholder.markdown(answer)
                # Asistan mesajını kaydet
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                error_msg = response.json().get("detail", "Bilinmeyen bir hata oluştu.")
                st.error(f"Sunucu Hatası: {error_msg}")
                message_placeholder.empty()
                
        except Exception as e:
            st.error(f"Hata: Sunucuya ulaşılamadı. \n{e}")
            message_placeholder.empty()