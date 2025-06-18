"""
This module handles Chroma DB initialization and RAG (retrieval-augmented generation)
functionality for the NutriChatBot.
"""

import sys
# --- Force pysqlite3 usage before chromadb import ---
# This is a common workaround for systems with older default sqlite3 libraries
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # If pysqlite3 is not available (e.g., local dev without it),
    # then fallback to the default sqlite3.
    # On Streamlit Cloud, pysqlite3-binary should be installed, so this won't be hit.
    pass
# --- End of pysqlite3 force ---


import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv() 
try:
    _gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise EnvironmentError("GEMINI_API_KEY not found. Please set it in your .env file or environment variables.")


def get_chroma_client():
    """
    Initializes and returns a Chroma DB client.
    The client uses DuckDB and stores data in the 'chroma/' directory.
    """
    client = chromadb.PersistentClient(path="./chroma")
    return client

def index_nutrition_data(df: pd.DataFrame, client: chromadb.PersistentClient):
    """
    Reads nutrition descriptions from a DataFrame and indexes them into a 'nutrition' collection
    in Chroma DB.
    """
    collection_name = "nutrition"
    
    # Check if collection exists and delete if it does to re-index
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass # Collection does not exist, safe to create

    # collection = client.get_or_create_collection(
    #     name=collection_name,
    #     embedding_function=_gemini_ef
    # )

    documents = df['description'].tolist() # Assuming 'description' column holds the text to embed
    metadatas = df.drop(columns=['description']).to_dict(orient='records')
    ids = [f"doc_{i}" for i in range(len(documents))]

    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    print(f"Indexed {len(documents)} nutrition items into Chroma DB.")

def rag_query(gemini_model: genai.GenerativeModel, query: str, client: chromadb.PersistentClient, n_results: int = 3):
    """
    Performs a RAG query:
    1. Retrieves top-N semantically similar documents from Chroma DB.
    2. Builds a prompt with the retrieved context.
    3. Calls gemini.chat.complete(...) with specified parameters.
    4. Returns the generated answer.
    """
    collection = client.get_collection(name="nutrition", embedding_function=_gemini_ef)
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Extract relevant documents and build context
    context = ""
    if results and results['documents']:
        for doc_list in results['documents']:
            for doc in doc_list:
                context += f"- {doc}\n"

    if context:
        prompt = f"""
        Anda adalah asisten gizi dan resep sehat. Gunakan informasi berikut sebagai konteks untuk menjawab pertanyaan pengguna.
        Jika informasi yang diberikan tidak relevan atau tidak cukup untuk menjawab pertanyaan, nyatakan bahwa Anda tidak memiliki informasi yang cukup.

        **Konteks Nutrisi:**
        {context}

        **Pertanyaan Pengguna:**
        {query}

        **Jawaban:**
        """
    else:
        prompt = f"""
        Anda adalah asisten gizi dan resep sehat. Saya tidak dapat menemukan informasi relevan berdasarkan pertanyaan Anda.
        Pertanyaan Anda: {query}
        Apakah ada hal lain yang bisa saya bantu?
        """

    chat = gemini_model.start_chat(history=[])
    response = chat.send_message(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 512})
    return response.text