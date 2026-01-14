import streamlit as st
import pdfplumber
import docx
import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client
import speech_recognition as sr  # <-- new import for STT

# ---------------- ENV ----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = genai.Client()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="🤖")


# ---------------- THEME & STYLING ----------------
# ---------------- THEME & STYLING ----------------
st.markdown("""
<style>
/* ---------------- GENERAL SETTINGS ---------------- */
@import url('https://fonts.googleapis.com/css2?family=Header:wght@400;600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #000000; 
    color: #FFFFFF;
}

/* Hiding Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ---------------- SIDEBAR ---------------- */
[data-testid="stSidebar"] {
    background-color: #050505; /* Almost black */
    border-right: 1px solid #1F1F1F;
}

[data-testid="stSidebar"] h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #FFFFFF;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
    text-align: left;
    background-color: transparent;
    color: #E0E0E0;
    border: 1px solid #333333;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease-in-out;
}

[data-testid="stSidebar"] .stButton button:hover {
    background-color: #1A1A1A;
    color: #FFFFFF;
    border-color: #58A6FF;
    transform: translateX(2px);
}

/* ---------------- CHAT INTERFACE ---------------- */

/* Input Container */
.stChatInputContainer {
    padding-bottom: 2rem;
    background-color: #000000;
}

.stChatInputContainer textarea {
    background-color: #0A0A0A !important; 
    color: #FFFFFF !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-size: 1rem;
    box-shadow: none;
}

.stChatInputContainer textarea:focus {
    border-color: #58A6FF !important;
    outline: none !important;
    box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.5);
    background-color: #111111 !important;
}

/* ---------------- BUTTONS & ELEMENTS ---------------- */
.stButton button {
    border-radius: 8px;
    font-weight: 600;
    background-color: #0E0E0E;
    color: #FFF;
    border: 1px solid #333;
}
.stButton button:hover {
    background-color: #1F1F1F;
    border-color: #555;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #FFFFFF !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #000000;
}
::-webkit-scrollbar-thumb {
    background: #333333;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #555555;
}

/* Welcome Message Styling */
.welcome-container {
    text-align: center;
    margin-top: 25vh;
    padding: 2rem;
    animation: fadeIn 1.2s ease-in-out;
}
.welcome-title {
    font-size: 3rem;
    color: #FFFFFF;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
}
.welcome-subtitle {
    font-size: 1.1rem;
    color: #666666;
    font-weight: 400;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

</style>
""", unsafe_allow_html=True)


# ---------------- SESSION ----------------
if "chat_id" not in st.session_state: st.session_state.chat_id = None
if "messages" not in st.session_state: st.session_state.messages = []
if "chats" not in st.session_state: st.session_state.chats = []

# ---------------- DB FUNCTIONS ----------------
def create_new_chat(title="New Chat"):
    res = supabase.table("chats").insert({"title": title}).execute()
    return res.data[0]["id"]

def load_chats():
    chats = supabase.table("chats").select("*").order("created_at", desc=True).execute().data
    st.session_state.chats = chats
    return chats

def load_messages(chat_id):
    return supabase.table("messages").select("*").eq("chat_id", chat_id).order("created_at").execute().data

def save_message(chat_id, role, content):
    supabase.table("messages").insert({"chat_id": chat_id, "role": role, "content": content}).execute()

def update_chat_title(chat_id, title):
    supabase.table("chats").update({"title": title[:50]}).eq("id", chat_id).execute()

def delete_chat(chat_id):
    supabase.table("messages").delete().eq("chat_id", chat_id).execute()
    supabase.table("chats").delete().eq("id", chat_id).execute()
    if st.session_state.chat_id == chat_id:
        st.session_state.chat_id = None
        st.session_state.messages = []

# ---------------- RAG FUNCTIONS ----------------
def get_embedding(text):
    res = client.models.embed_content(model="text-embedding-004", contents=text)
    return res.embeddings[0].values

def save_document(text, chat_id):
    emb = get_embedding(text)
    supabase.table("documents").insert({"content": text, "embedding": emb, "chat_id": chat_id}).execute()

def fetch_similar_docs(query, chat_id):
    q_emb = get_embedding(query)
    try:
        res = supabase.rpc("match_documents", {
            "query_embedding": q_emb,
            "match_count": 5,
            "match_threshold": 0.3,
            "chat_id_filter": chat_id
        }).execute()
    except Exception:
        return ""
    if not res.data:
        return ""
    return "\n\n".join([r["content"] for r in res.data])

def rag_answer(question, history, chat_id):
    docs = fetch_similar_docs(question, chat_id)
    context = "".join([f"{m['role']}: {m['content']}\n" for m in history[-4:]])
    prompt = f"""
    You are a helpful assistant that uses both uploaded documents and general knowledge to answer questions.

    Instructions:
    1. **PRIORITIZE** information from the provided 'Documents' (Source of Truth).
    2. If the answer is found in the 'Documents', cite or use that information clearly.
    3. If the answer is **NOT** in the 'Documents', you may use your general knowledge to answer the question.
    4. Explicitly mention if you are using general knowledge vs document knowledge if it's ambiguous.
    5. Maintain the flow of the conversation based on the 'Conversation context'.

    Conversation context:
    {context if context else 'None'}

    Documents (Source of Truth):
    {docs if docs else 'No relevant documents found.'}

    Question: {question}
    """
    res = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
    return res.text

# ---------------- SPEECH-TO-TEXT FUNCTION ----------------
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Please speak now.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I could not understand your speech."
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition."

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("Chatbot")
    
    if st.button("➕ New Chat", key="new_chat"):
        st.session_state.chat_id = create_new_chat()
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    
    chats = load_chats()
    for chat in chats:
        cols = st.columns([0.85, 0.15])
        if cols[0].button(chat["title"] or "Untitled", key=f"select_{chat['id']}"):
            st.session_state.chat_id = chat["id"]
            st.session_state.messages = load_messages(chat["id"])
            st.rerun()
        if cols[1].button("🗑️", key=f"delete_{chat['id']}"):
            delete_chat(chat["id"])
            st.rerun()

    st.markdown("---")
    st.caption("📂 **Upload Context**")
    uploaded_file = st.file_uploader("Attach PDF/TXT/DOCX", type=["pdf", "txt", "docx", "doc"], label_visibility="collapsed")
    
    if uploaded_file:
        if not st.session_state.chat_id:
            st.session_state.chat_id = create_new_chat()
            st.session_state.messages = []

        text = ""
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for p in pdf.pages:
                    if p.extract_text():
                        text += p.extract_text() + "\n"
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
             save_document(text, st.session_state.chat_id)
             doc_msg = f"📄 Uploaded: {uploaded_file.name}"
             save_message(st.session_state.chat_id, "assistant", doc_msg)
             st.session_state.messages.append({"role": "assistant", "content": doc_msg})
             st.session_state.last_uploaded = uploaded_file.name
             st.success("File processed!")

    # ---------------- Speech-to-Text Button ----------------
    st.markdown("🎤 **Or speak your question:**")
    if st.button("Speak"):
        if not st.session_state.chat_id:
            st.session_state.chat_id = create_new_chat()
        user_query = speech_to_text()
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.markdown(user_query)
        save_message(st.session_state.chat_id, "user", user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_answer(user_query, st.session_state.messages, st.session_state.chat_id)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_message(st.session_state.chat_id, "assistant", response)

# ---------------- MAIN PANEL ----------------
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <h1 class="welcome-title">RAG Chatbot</h1>
        <p class="welcome-subtitle">Ask me anything or upload a document to get started.</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Message ChatGPT..."):
    if not st.session_state.chat_id:
        st.session_state.chat_id = create_new_chat()
        
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    if len(st.session_state.messages) == 1:
         update_chat_title(st.session_state.chat_id, prompt)
    save_message(st.session_state.chat_id, "user", prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
             response = rag_answer(prompt, st.session_state.messages, st.session_state.chat_id)
             st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.chat_id, "assistant", response)
