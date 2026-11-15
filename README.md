# NutriChatBot

## Overview

**NutriChatBot** adalah chatbot web interaktif berbasis **Streamlit** dan **SumoPod AI** (OpenAI-compatible) yang membantu pengguna:

- Menjawab pertanyaan gizi secara akurat dengan **RAG (Retrieval-Augmented Generation)**
- Merekomendasikan resep sehat berdasarkan bahan yang tersedia
- Menyusun rencana menu mingguan sesuai target kalori
- Menyediakan alternatif bahan makanan yang lebih sehat

Tujuan utama: memberdayakan pengguna membuat pilihan makanan sehat, menghemat waktu, dan meminimalkan limbah makanan.

## Features

1. **💬 Tanya Gizi (Hybrid RAG + Conversational Memory)**:
   - Pencarian cerdas **Hybrid Search (Reciprocal Rank Fusion)**: Menggabungkan pencarian leksikal teks persis (SQLite) dan pencarian semantik vektor (ChromaDB + SumoPod `text-embedding-3-small`).
   - Basis data nutrisi transparan: Standar **100 gram porsi dapat dimakan (BDD)**.
   - Respon **Streaming real-time** kata per kata.
   - Riwayat percakapan interaktif (*multi-turn conversation*).
2. **🍳 Masak Apa Hari Ini? (Ide Resep Sehat)**:
   - Rekomendasi resep bergizi lengkap dengan takaran dan langkah memasak berdasarkan bahan yang ada.
   - Pilihan filter diet: *Tinggi Protein*, *Rendah Garam*, *Rendah Gula*, *Vegetarian*, *Vegan*, *Bebas Gluten*.
   - Fitur unduh resep dalam format Markdown (`.md`).
3. **📅 Perencana Menu Mingguan & Kalkulator Gizi Personal**:
   - Menu makan 7 hari (Senin - Minggu) seimbang sesuai target kalori.
   - **Kalkulator BMR & TDEE (Mifflin-St Jeor)** terintegrasi untuk menghitung kebutuhan kalori dan rekomendasi makronutrisi (Protein, Lemak, Karbohidrat) secara personal.
   - Fitur unduh rencana menu 7 hari dalam format Markdown (`.md`).
4. **📊 Bandingkan Makanan (Visual Nutrition Comparison)**:
   - Komparasi nilai gizi antara 2 hidangan/makanan secara berdampingan.
   - Grafik batang visual (*interactive bar chart*) untuk perbandingan makronutrisi.
   - Thumbnail foto makanan dari dataset asli.
5. **🥗 Alternatif Sehat**:
   - Saran substitusi bahan pangan yang lebih bergizi dan cara penggunaannya.
6. **🛡️ Medical Safety Guardrails & Disclaimer**:
   - Deteksi otomatis kondisi medis berisiko (diabetes, hipertensi, ginjal, asam urat, alergi) untuk memberikan peringatan klinis khusus.

## Architecture & Project Structure

```text
[User] → Streamlit UI (app.py)
       → Core Services:
           ├── services/database.py       (SQLite Lexical Keyword Search & CSV Validation)
           ├── services/embeddings.py     (Hybrid RRF: Lexical + ChromaDB Dense Vector)
           ├── services/llm.py            (Streaming GLM-5.3-flash, Guardrails & Disclaimer)
           └── services/nutrition_calc.py (BMR, TDEE, & Macro Distribution Formula)
       → Data Stores:
           ├── SQLite (nutrition.db, 1.346 records)
           └── ChromaDB (/chroma, HNSW Vector Index)
       → External APIs:
           └── SumoPod AI (https://ai.sumopod.com/v1 - GLM & text-embedding-3-small)
```

```text
NutriChatBot/
├── app.py                      # Streamlit UI entrypoint (Streaming, Chat & Visualization)
├── config.py                   # Centralized env config, constants, & redacted logger
├── embeddings.py               # Backward-compatible facade for RAG helpers
├── services/
│   ├── database.py             # SQLite connection manager, keyword ranking & CSV validation
│   ├── embeddings.py           # ChromaDB client, Hybrid RRF retrieval & batch indexing
│   ├── llm.py                  # LLM streaming builders, clinical guardrails & OpenAI client
│   └── nutrition_calc.py       # Biometric calculations (BMR, TDEE, Mifflin-St Jeor)
├── scripts/
│   └── rebuild_index.py        # CLI tool to rebuild SQLite & ChromaDB vector index
├── tests/                      # Comprehensive automated pytest suite (40+ tests)
├── data/
│   └── nutrition.csv           # Indonesian food nutrition dataset (1,346 items)
├── Dockerfile                  # Production multi-stage Docker container
├── .dockerignore               # Docker ignore rules
├── .github/workflows/ci.yml    # Automated CI workflow (Ruff Lint & Pytest)
├── .env.example                # Environment variables template
└── pyproject.toml              # Project metadata & dependencies
```

### Technology Stack

- **Bahasa**: Python 3.10+
- **Front-end**: Streamlit
- **AI/LLM**: SumoPod AI (`https://ai.sumopod.com/v1`) via `openai` SDK
  - LLM Model: `glm-5.3-flash` (dapat diubah via `MODEL_LLM`)
  - Embedding Model: `text-embedding-3-small` (dapat diubah via `MODEL_EMBEDDING`)
- **Vector DB**: ChromaDB (SQLite metadata + HNSW vector index)
- **Relational DB**: SQLite
- **Dependency & Env Management**: `uv`, `python-dotenv`
- **Testing & Linting**: `pytest`, `ruff`

## Installation & Setup

### Prasyarat

- Python 3.10+
- [uv](https://astral.sh/uv)
- API key SumoPod AI

### Langkah Instalasi

```bash
git clone https://github.com/victoriavals/NutriChatBot.git
cd NutriChatBot

# Buat virtual environment & install seluruh dependencies (termasuk dev tools)
uv venv
uv sync --extra dev

# Siapkan environment variables
cp .env.example .env
# Edit .env dan isi API_KEY dengan API key SumoPod aktif Anda
```

### Konfigurasi `.env`

```env
# Required: SumoPod AI / OpenAI-compatible API Key
API_KEY=your_api_key_here

# Optional: Base URL (default: https://ai.sumopod.com/v1)
BASE_URL=https://ai.sumopod.com/v1

# Models configuration
MODEL_LLM=glm-5.3-flash
MODEL_EMBEDDING=text-embedding-3-small
```

> [!NOTE]
> `pyproject.toml` sudah menggunakan marker `; sys_platform == 'linux'` untuk `pysqlite3-binary`, sehingga instalasi lokal di Windows/macOS maupun deployment di Linux/Streamlit Cloud berjalan otomatis tanpa perlu mengedit `pyproject.toml` secara manual.

### Membangun Ulang Index Database & Vector (RAG)

`nutrition.db` akan otomatis diinisialisasi dari `data/nutrition.csv` saat aplikasi berjalan, sedangkan index vektor **ChromaDB** untuk fitur **Tanya Gizi (RAG)** dapat dibangun melalui CLI atau menu **Index Data** di aplikasi:

```bash
# Bangun ulang SQLite dan ChromaDB sekaligus membersihkan segment lama/yatim
uv run python scripts/rebuild_index.py --clean

# Atau hanya bangun ulang SQLite tanpa memanggil Embeddings API
uv run python scripts/rebuild_index.py --skip-chroma
```

### Menjalankan Aplikasi

```bash
uv run streamlit run app.py
```

### Menjalankan via Docker

```bash
# Build Docker image
docker build -t nutrichatbot:latest .

# Jalankan container
docker run -d -p 8501:8501 --env-file .env --name nutrichatbot-app nutrichatbot:latest
```
Akses aplikasi melalui browser di `http://localhost:8501`.


## Testing & Verification

```bash
# Jalankan seluruh automated tests (offline dengan mock untuk LLM & ChromaDB)
uv run pytest -v

# Syntax & lint check
uv run python -m py_compile app.py config.py embeddings.py services/*.py scripts/*.py
uv run ruff check .
```

## Usage

- **Tanya Gizi**: `Berapa kalori sepiring nasi goreng?` (memiliki fallback pencarian SQLite apabila index ChromaDB belum dibangun)
- **Masak Apa Hari Ini?**: `ayam, brokoli, bawang putih`
- **Perencana Menu Mingguan**: masukkan target kalori harian (`1000–5000` kcal), klik *Buat Rencana Menu*
- **Alternatif Sehat**: `gula` → substitusi sehat beserta alasan dan cara penggunaan
- **Index Data**: muat ulang CSV dan indeks ulang ChromaDB langsung dari UI

## Demo

Akses demo: `https://nutrichatbot.streamlit.app/`

## License & Credits

- **License**: MIT
- **Data nutrisi** diambil dari Kaggle dataset: [Indonesian Food and Drink Nutrition Dataset](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset)
- **Dikembangkan oleh**: Naufal Firdaus
