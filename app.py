"""
This is the main Streamlit application for NutriChatBot: Asisten Gizi & Resep Sehat.
It provides various functionalities including RAG-based nutrition questions,
recipe generation, weekly meal planning, and healthy alternatives.
"""

import streamlit as st
import pandas as pd
import sqlite3
import os
import google.generativeai as genai
from dotenv import load_dotenv
from embeddings import get_chroma_client, index_nutrition_data, rag_query

# --- Environment Loading and Gemini Client Initialization ---
load_dotenv()
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
        st.stop()
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    st.error(f"Error initializing Gemini: {e}")
    st.stop()

# --- SQLite Initialization ---
DB_NAME = "nutrition.db"
CSV_PATH = "data/nutrition.csv"

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    return conn

def load_data(conn):
    """
    Reads nutrition data from data/nutrition.csv, normalizes column names,
    stores it into an SQLite table 'nutrition', and then indexes it into Chroma DB.
    """
    st.info(f"Loading data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        # Normalize column names by making them lowercase and replacing spaces with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        # Ensure 'description' column exists for RAG indexing
        if 'name' in df.columns and 'calories' in df.columns:
            df['description'] = df['name'] + ' has ' + df['calories'].astype(str) + ' calories, ' + \
                                df['proteins'].astype(str) + 'g protein, ' + \
                                df['fat'].astype(str) + 'g fat, and ' + \
                                df['carbohydrate'].astype(str) + 'g carbohydrate.'
        else:
            st.error("CSV must contain 'name' and 'calories' columns for description generation.")
            return

        # Store into SQLite
        df.to_sql('nutrition', conn, if_exists='replace', index=False)
        st.success("Nutrition data loaded into SQLite database.")

        # Index into Chroma DB
        chroma_client = get_chroma_client()
        index_nutrition_data(df, chroma_client)
        st.success("Nutrition data indexed into Chroma DB for RAG.")

    except FileNotFoundError:
        st.error(f"Error: {CSV_PATH} not found. Please make sure it's in the 'data/' directory.")
    except Exception as e:
        st.error(f"Error loading and indexing data: {e}")

def get_nutrition(conn, item: str):
    """Queries SQLite for nutrition information about a specific item."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM nutrition WHERE LOWER(name) LIKE ?", ('%' + item.lower() + '%',))
    return cursor.fetchall()

# --- Streamlit UI ---
st.set_page_config(
    page_title="NutriChatBot: Asisten Gizi & Resep Sehat",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("NutriChatBot: Asisten Gizi & Resep Sehat 🍎🥕")
st.markdown("Asisten pribadi Anda untuk nutrisi dan resep sehat.")

with st.sidebar:
    st.header("Menu")
    menu_selection = st.radio("Pilih Opsi", [
        # "Index Data", #UNCOMMENT THIS LINE TO BUILD CHROMA DB (RAG)
        "Tanya Gizi",
        "Masak Apa Hari Ini?",
        "Perencana Menu Mingguan",
        "Alternatif Sehat"
    ])
    st.markdown("---")
    st.write("By Naufal Firdaus")

conn = get_db_connection()

if menu_selection == "Index Data":
    st.header("⚙️ Index Data Nutrisi")
    st.write("Klik tombol di bawah untuk memuat data nutrisi dari `data/nutrition.csv` ke dalam database SQLite dan mengindeksnya ke Chroma DB.")
    if st.button("Muat & Indeks Data"):
        load_data(conn)
        conn.close()

elif menu_selection == "Tanya Gizi":
    st.header("💬 Tanya Gizi")
    st.write("Data gizi diambil berdasarkan data yang kami miliki.")
    st.write("Ajukan pertanyaan tentang gizi makanan yang sudah diindeks. Contoh: 'Berapa kalori nasi goreng?'")
    user_query = st.text_input("Pertanyaan Anda:", key="rag_query_input")
    if st.button("Cari Jawaban", key="rag_query_button"):
        if user_query:
            with st.spinner("Mencari jawaban..."):
                try:
                    chroma_client = get_chroma_client()
                    answer = rag_query(gemini_model, user_query, chroma_client)
                    st.write("---")
                    st.subheader("Jawaban:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses pertanyaan RAG: {e}")
        else:
            st.warning("Mohon masukkan pertanyaan Anda.")

elif menu_selection == "Masak Apa Hari Ini?":
    st.header("🍳 Masak Apa Hari Ini?")
    st.write("Berikan daftar bahan-bahan yang Anda miliki (dipisahkan koma), dan saya akan memberikan ide resep.")
    ingredients = st.text_input("Bahan-bahan (contoh: ayam, brokoli, bawang putih):", key="recipe_ingredients")
    if st.button("Dapatkan Resep", key="get_recipe_button"):
        if ingredients:
            with st.spinner("Mencari resep..."):
                recipe_prompt = f"""
                Buatkan saya ide resep makanan lengkap berdasarkan bahan-bahan berikut: {ingredients}.
                Sertakan nama resep, bahan-bahan lengkap (termasuk yang tidak disebutkan jika umum),
                dan langkah-langkah pembuatannya yang jelas.
                Format jawaban Anda dalam bahasa Indonesia yang mudah dipahami.
                """
                try:
                    chat = gemini_model.start_chat(history=[])
                    response = chat.send_message(recipe_prompt, generation_config={"temperature": 0.3, "max_output_tokens": 512})
                    st.write("---")
                    st.subheader("Ide Resep Anda:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membuat resep: {e}")
        else:
            st.warning("Mohon masukkan bahan-bahan yang Anda miliki.")

elif menu_selection == "Perencana Menu Mingguan":
    st.header("📅 Perencana Menu Mingguan")
    st.write("Saya akan membantu Anda membuat rencana menu mingguan berdasarkan kebutuhan kalori harian Anda.")
    daily_calories = st.number_input("Target Kalori Harian (kcal):", min_value=1000, max_value=5000, value=2000, step=100, key="daily_calories")
    if st.button("Buat Rencana Menu", key="generate_menu_button"):
        if daily_calories:
            with st.spinner("Membuat rencana menu..."):
                menu_prompt = f"""
                Buatkan rencana menu makanan sehat untuk satu minggu (7 hari) dengan target rata-rata {daily_calories} kalori per hari.
                Sertakan sarapan, makan siang, makan malam, dan dua camilan.
                Berikan ide makanan yang bervariasi dan seimbang secara nutrisi.
                Format jawaban Anda dalam bahasa Indonesia dengan struktur per hari (contoh: Senin: Sarapan, Makan Siang, dst.).
                """
                try:
                    chat = gemini_model.start_chat(history=[])
                    response = chat.send_message(menu_prompt, generation_config={"temperature": 0.2, "max_output_tokens": 2048})
                    st.write("---")
                    st.subheader("Rencana Menu Mingguan Anda:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat membuat rencana menu: {e}")
        else:
            st.warning("Mohon masukkan target kalori harian Anda.")

elif menu_selection == "Alternatif Sehat":
    st.header("🥗 Alternatif Sehat")
    st.write("Masukkan bahan makanan yang ingin Anda cari alternatif sehatnya.")
    ingredient_to_replace = st.text_input("Bahan yang ingin diganti (contoh: gula, mentega):", key="healthy_alternative_input")
    if st.button("Cari Alternatif", key="find_alternative_button"):
        if ingredient_to_replace:
            with st.spinner("Mencari alternatif..."):
                alternative_prompt = f"""
                Berikan beberapa alternatif sehat untuk bahan makanan '{ingredient_to_replace}'.
                Sertakan mengapa alternatif tersebut lebih sehat dan bagaimana cara menggunakannya.
                Format jawaban Anda dalam bahasa Indonesia.
                """
                try:
                    chat = gemini_model.start_chat(history=[])
                    response = chat.send_message(alternative_prompt, generation_config={"temperature": 0.5, "max_output_tokens": 256})
                    st.write("---")
                    st.subheader(f"Alternatif Sehat untuk '{ingredient_to_replace}':")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mencari alternatif sehat: {e}")
        else:
            st.warning("Mohon masukkan bahan yang ingin Anda ganti.")

conn.close()