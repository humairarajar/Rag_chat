import streamlit as st
import pdfplumber
import docx
import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client

# ---------------- ENV ----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = genai.Client()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="🤖")

# ---------------- THEME & STYLING ----------------
st.markdown("""
<style>
/* Main Background */
.stApp {
    background-color: #FFFFFF;
    color: #333333;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #F7F7F8;
    border-right: 1px solid #ECECEC;
}

/* Chat Input styling - Simple Light */
.stChatInputContainer textarea {
    background-color: #FFFFFF !important;
    color: #333333 !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #FFFFFF; 
}
::-webkit-scrollbar-thumb {
    background: #D1D5DB; 
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #9CA3AF; 
}

/* Buttons in sidebar */
.stButton button {
    background-color: transparent;
    color: #333333;
    border: 1px solid #E5E5E5;
    border-radius: 6px;
    transition: all 0.2s ease;
    width: 100%;
    text-align: left;
    padding-left: 10px;
}
.stButton button:hover {
    background-color: #E5E5E5;
    border-color: #D1D5DB;
    color: #000;
}

/* Remove default main menu and footer for immersion */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

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
            "match_threshold": 0.3, # Lowered to capture more relevant context
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
        # Using a unique key for buttons
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
        
        # Process and save if new
        # Note: A smarter way is to check if we already processed this file in this session, 
        # but for now we'll just process it when it appears. 
        # To avoid re-processing on every rerun, we can check a session state flag or just let the user know.
        # Ideally, we'd clear the uploader after success, but Streamlit makes that tricky without a form.
        # We will just process and show a toast.
        
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
        
        # Simple check to avoid spamming the DB on every rerun if the file sits there
        # For this demo, we'll assume the user removes it or ignores the re-upload logic, 
        # OR we can store the last_uploaded_filename in session_state.
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
             save_document(text, st.session_state.chat_id)
             doc_msg = f"📄 Uploaded: {uploaded_file.name}"
             save_message(st.session_state.chat_id, "assistant", doc_msg)
             st.session_state.messages.append({"role": "assistant", "content": doc_msg})
             st.session_state.last_uploaded = uploaded_file.name
             st.success("File processed!")

# ---------------- MAIN PANEL ----------------
# No huge header, just the chat
if not st.session_state.messages:
    # Show a welcome screen if empty
    st.markdown("""
    <div style="text-align: center; margin-top: 20vh; color: #666;">
        <h1>How can I help you today?</h1>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- CHAT INPUT ----------------
if prompt := st.chat_input("Message ChatGPT..."):
    if not st.session_state.chat_id:
        st.session_state.chat_id = create_new_chat()
        
    # Optimistic UI: Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Save user message
    if len(st.session_state.messages) == 1:
         update_chat_title(st.session_state.chat_id, prompt)
    save_message(st.session_state.chat_id, "user", prompt)

    # Generate Response
    with st.chat_message("assistant"):
        # Placeholder or spinner
        with st.spinner("Thinking..."):
             response = rag_answer(prompt, st.session_state.messages, st.session_state.chat_id)
             st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    save_message(st.session_state.chat_id, "assistant", response)

