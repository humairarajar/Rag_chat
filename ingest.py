import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = genai.Client()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_embedding(text):
    res = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return res.embeddings[0].values

def insert_document(text, chat_id):
    emb = get_embedding(text)
    supabase.table("documents").insert({
        "content": text,
        "embedding": emb,
        "chat_id": chat_id
    }).execute()

    print("Inserted for chat:", chat_id)
