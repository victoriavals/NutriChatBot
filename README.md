# NutriChatBot

## Project Title

**NutriChatBot**

## Overview

NutriChatBot adalah chatbot web interaktif yang membantu kamu:

- Menjawab pertanyaan gizi dengan akurat
- Merekomendasikan resep sehat berdasarkan bahan di kulkas
- Menyusun rencana menu mingguan sesuai target kalori
- Menyediakan alternatif bahan sehat

Tujuan utama: memberdayakan pengguna membuat pilihan makanan sehat, menghemat waktu, dan meminimalkan limbah makanan.

## Motivation

Di tengah banjir informasi gizi, sulit mendapatkan data yang terpercaya dan cepat. Banyak orang mengalami:

- Kebingungan tentang kandungan nutrisi makanan
- Tantangan merencanakan menu sehat harian atau mingguan
- Kendala memanfaatkan bahan sisa di rumah

**Target pengguna**: individu yang peduli kesehatan, sedang diet, maupun pencari inspirasi resep.

## Features

- **Tanya Gizi (RAG)**: jawaban gizi akurat dari basis data vektor (ChromaDB)
- **Masak Apa Hari Ini?**: ide resep otomatis dari bahan yang ada
- **Perencana Menu Mingguan**: susun menu 7 hari sesuai target kalori
- **Alternatif Sehat**: saran substitusi bahan lebih sehat

## Architecture & Technology Stack

```text
[User] → Streamlit (Front‑end) → Back‑end Python  
      → RAG Engine (ChromaDB + Google Gemini API)  
      → SQLite (data nutrisi & resep)
```

- **Bahasa**: Python 3.10+
- **Front‑end**: Streamlit
- **AI/ML**: Google Gemini API (chat & embedding)
- **Vector DB**: ChromaDB (DuckDB + Parquet)
- **Relational DB**: SQLite
- **Env**: python-dotenv, uv

## How It Works

1. **Input**: pengguna kirim pertanyaan atau daftar bahan
2. **Embedding & Retrieval**: query diubah ke vektor, cari top‑k di ChromaDB
3. **Prompt Engineering**: gabungkan konteks, kirim ke Gemini LLM
4. **Generation**: Gemini hasilkan jawaban/resep/menu
5. **Display**: tampilkan respons di Streamlit

## Installation & Setup

### Prasyarat

- Python 3.10+
- uv ([https://astral.sh/uv](https://astral.sh/uv))
- API key Google Gemini

### Langkah instalasi

```bash
git clone https://github.com/victoriavals/nutrichatbot.git
cd nutrichatbot

# buat virtual env & install deps
date
uv venv
uv sync

# tambahkan .env
echo "GEMINI_API_KEY=YOUR_KEY_HERE" > .env

# siapkan dataset
# file 'nutrition.csv' di folder data/ berasal dari Kaggle:
# https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset

buka file pyproject.toml lalu command pada baris "pysqlite3-binary" karena itu hanya untuk environment linux

# jalankan aplikasi
uv run streamlit run app.py
```

## Usage

- **Tanya Gizi**: `Berapa kalori sepiring nasi goreng?`
- **Resep**: `ayam, kentang, wortel`
- **Menu Mingguan**: masukkan target kalori, klik *Buat Rencana*
- **Alternatif**: `gula` → substitusi sehat

## Demo

Akses demo:

```
https://nutrichatbot.streamlit.app/
```

## Future Improvements

- Fine‑tuning lokal model dengan resep Indonesia
- Integrasi deteksi gambar makanan
- Profil pengguna untuk rekomendasi lebih personal
- Dukungan multi‑bahasa

## License & Credits

- **License**: MIT
- **Data nutrisi** diambil dari Kaggle dataset: [Indonesian Food and Drink Nutrition Dataset](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset)
- **Dikembangkan oleh**: [Naufal Firdaus]

