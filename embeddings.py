from chromadb import Client
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

def get_chroma_client():
    return Client(Settings(
        persist_directory="chroma/",
        chroma_db_impl="duckdb+parquet"
    ))

# Tambahkan dokumen deskripsi nutrisi ke Chroma
def index_nutrition_data(df):
    client = get_chroma_client()
    if "nutrition" not in [c.name for c in client.list_collections()]:
        client.create_collection(name="nutrition")
    col = client.get_collection(name="nutrition")
    docs = df["description"].fillna("").tolist()
    metas = df[["name"]].to_dict(orient="records")
    ids = [f"item_{i}" for i in range(len(docs))]
    col.add(documents=docs, ids=ids, metadatas=metas)

# Pencarian semantik
def rag_query(gemini, query: str, client: Client, n_results: int = 3) -> str:
    col = client.get_collection(name="nutrition")
    res = col.query(query_texts=[query], n_results=n_results)
    context = "\n".join(res["documents"][0])
    prompt = (
        f"Informasi nutrisi relevan:\n{context}\n\n"
        f"Jawab pertanyaan berikut berdasarkan konteks: {query}"
    )
    resp = gemini.chat.complete(
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=512
    )
    return resp.text