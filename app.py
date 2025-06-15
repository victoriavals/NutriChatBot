import os
import sqlite3
import pandas as pd
import streamlit as st
from google import genai
from dotenv import load_dotenv
from embeddings import get_chroma_client, index_nutrition_data, rag_query

# 1. Load .env & init clients
def load_env():
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        st.error("Set GEMINI_API_KEY di .env!")
        st.stop()
    return key

def init_db(path: str = "nutrition.db") -> sqlite3.Connection:
    return sqlite3.connect(path, check_same_thread=False)

# 2. Load nutrition.csv ke SQLite & Chroma (jalankan sekali)
def load_data(conn):
    df = pd.read_csv("data/nutrition.csv")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.to_sql("nutrition", conn, if_exists="replace", index=False)
    index_nutrition_data(df)
    st.success("Data nutrisi terindeks ke DB & Chroma.")

# 3. Query nutrisi via SQLite
def get_nutrition(conn, item: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM nutrition WHERE name LIKE ?", conn, params=(f"%{item}%",)
    )

# 4. UI Streamlit
def main():
    st.title("🍽️ NutriChatBot: Asisten Gizi & Resep Sehat")
    key = load_env()
    gemini = genai.Client(api_key=key)
    conn = init_db()
    client = get_chroma_client()

    menu = st.sidebar.selectbox("Pilih Fitur", [
        "Index Data",
        "Tanya Gizi (RAG)",
        "Masak Apa Hari Ini?",
        "Perencana Menu Mingguan",
        "Alternatif Sehat"
    ])

    if menu == "Index Data":
        if st.button("Load & Index Data"):
            load_data(conn)

    elif menu == "Tanya Gizi (RAG)":
        q = st.text_input("Pertanyaan nutrisi:")
        if st.button("Cari Jawaban") and q:
            with st.spinner("Mencari jawaban..."):
                ans = rag_query(gemini, q, client)
            st.markdown(ans)

    elif menu == "Masak Apa Hari Ini?":
        raw = st.text_input("Bahan (pisah koma):")
        if st.button("Cari Resep") and raw:
            ing = [i.strip() for i in raw.split(',')]
            prompt = ("Buat resep sederhana dengan: " + ', '.join(ing))
            res = gemini.chat.complete(prompt=prompt, temperature=0.3, max_output_tokens=512)
            st.markdown(res.text)

    elif menu == "Perencana Menu Mingguan":
        cal = st.number_input("Kalori harian:", 1000, 4000, 2000)
        if st.button("Buat Rencana"):
            prmpt = (f"Rencanakan menu mingguan ~{cal} kcal/hari...")
            res = gemini.chat.complete(prompt=prmpt, temperature=0.2, max_output_tokens=512)
            st.markdown(res.text)

    else:  # Alternatif Sehat
        ing = st.text_input("Bahan ganti:")
        if st.button("Cari Alternatif") and ing:
            prmpt = f"Alternatif sehat untuk '{ing}' dan kelebihan gizinya."
            res = gemini.chat.complete(prompt=prmpt, temperature=0.5, max_output_tokens=256)
            st.markdown(res.text)

if __name__ == "__main__":
    main()