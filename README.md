# NutriChatBot

## Overview

**NutriChatBot** adalah chatbot web interaktif berbasis **Streamlit** dan **Google Gemini API** yang membantu pengguna:

- Menjawab pertanyaan gizi secara akurat dengan **RAG (Retrieval-Augmented Generation)**
- Merekomendasikan resep sehat berdasarkan bahan yang tersedia
- Menyusun rencana menu mingguan sesuai target kalori
- Menyediakan alternatif bahan makanan yang lebih sehat

Tujuan utama: memberdayakan pengguna membuat pilihan makanan sehat, menghemat waktu, dan meminimalkan limbah makanan.

## Architecture & Project Structure

```text
[User] → Streamlit UI (app.py)
       → Service Layer (services/llm.py, services/database.py, services/embeddings.py)
       → RAG Engine (ChromaDB SQLite + HNSW + Google Gemini API)
       → SQLite (nutrition.db, auto-initialized dari data/nutrition.csv)
```

```text
NutriChatBot/
├── app.py                      # Streamlit UI entrypoint (with @st.cache_resource)
├── config.py                   # Centralized env config, constants, & redacted logger
├── embeddings.py               # Backward-compatible facade for RAG helpers
├── services/
│   ├── database.py             # SQLite connection manager, CSV validation & search
│   ├── embeddings.py           # ChromaDB client, batched indexing, & retrieval
│   └── llm.py                  # Gemini prompt builders & text generation
├── scripts/
│   └── rebuild_index.py        # CLI tool to rebuild SQLite & ChromaDB index
├── tests/                      # Automated pytest suite
├── data/
│   └── nutrition.csv           # Indonesian food nutrition dataset (1,346 items)
├── .env.example                # Environment variables template
└── pyproject.toml              # Project metadata & PEP 508 dependencies
```

### Technology Stack

- **Bahasa**: Python 3.10+
- **Front-end**: Streamlit
- **AI/LLM**: Google Gemini API (`gemini-2.0-flash` default, dapat diubah via `GEMINI_MODEL`)
- **Vector DB**: ChromaDB (SQLite metadata + HNSW vector index)
- **Relational DB**: SQLite
- **Dependency & Env Management**: `uv`, `python-dotenv`
- **Testing & Linting**: `pytest`, `ruff`

## Installation & Setup

### Prasyarat

- Python 3.10+
- [uv](https://astral.sh/uv)
- API key Google Gemini dari Google AI Studio

### Langkah Instalasi

```bash
git clone https://github.com/victoriavals/NutriChatBot.git
cd NutriChatBot

# Buat virtual environment & install seluruh dependencies (termasuk dev tools)
uv venv
uv sync --extra dev

# Siapkan environment variables
cp .env.example .env
# Edit .env dan isi GEMINI_API_KEY dengan API key aktif Anda
```

> [!NOTE]
> `pyproject.toml` sudah menggunakan marker `; sys_platform == 'linux'` untuk `pysqlite3-binary`, sehingga instalasi lokal di Windows/macOS maupun deployment di Linux/Streamlit Cloud berjalan otomatis tanpa perlu mengedit `pyproject.toml` secara manual.

### Membangun Ulang Index Database & Vector (RAG)

`nutrition.db` akan otomatis diinisialisasi dari `data/nutrition.csv` saat aplikasi berjalan, sedangkan index vektor **ChromaDB** untuk fitur **Tanya Gizi (RAG)** dapat dibangun melalui CLI atau menu **Index Data** di aplikasi:

```bash
# Bangun ulang SQLite dan ChromaDB sekaligus membersihkan segment lama/yatim
uv run python scripts/rebuild_index.py --clean

# Atau hanya bangun ulang SQLite tanpa memanggil Gemini Embeddings API
uv run python scripts/rebuild_index.py --skip-chroma
```

### Menjalankan Aplikasi

```bash
uv run streamlit run app.py
```

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
