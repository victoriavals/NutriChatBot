# NutriChatBot 🍎🥕

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nutrichatbot.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![AI Provider](https://img.shields.io/badge/AI-SumoPod%20AI%20%2F%20GLM--5.3--flash-orange.svg)](https://ai.sumopod.com/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-ChromaDB-green.svg)](https://www.trychroma.com/)
[![Database](https://img.shields.io/badge/Database-SQLite3-003B57.svg)](https://www.sqlite.org/)
[![Tests](https://img.shields.io/badge/tests-51%2F51%20passing-brightgreen.svg)](tests/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Asisten Nutrisi, Pola Makan Sehat, dan Rekomendasi Menu Cerdas Nusantara Berbasis Dual-Engine Hybrid RAG & Mifflin-St Jeor.**  
> 🌐 **Live Demo Publik**: [https://nutrichatbot.streamlit.app/](https://nutrichatbot.streamlit.app/)

---

## 📌 Daftar Isi

1. [Ikhtisar Proyek (Overview)](#-ikhtisar-proyek-overview)
2. [Fitur Unggulan (Key Features)](#-fitur-unggulan-key-features)
3. [Arsitektur & Landasan Rekayasa](#-arsitektur--landasan-rekayasa)
   - [Dual-Engine Hybrid Retrieval (wRRF)](#1-dual-engine-hybrid-retrieval-wrrf)
   - [Kalkulator Metabolik Klinis (Mifflin-St Jeor)](#2-kalkulator-metabolik-klinis-mifflin-st-jeor)
   - [Guardrail Keselamatan Medis & Redaksi Data](#3-guardrail-keselamatan-medis--redaksi-data)
   - [User-Centered Design (UCD) untuk Pengguna Awam](#4-user-centered-design-ucd-untuk-pengguna-awam)
4. [Struktur Direktori Proyek](#-struktur-direktori-proyek)
5. [Tech Stack](#-tech-stack)
6. [Panduan Instalasi & Menjalankan](#-panduan-instalasi--menjalankan)
   - [Prasyarat](#prasyarat)
   - [Langkah Setup dengan `uv`](#langkah-setup-dengan-uv-direkomendasikan)
   - [Konfigurasi Environment (.env)](#konfigurasi-environment-env)
   - [Membangun Ulang Database & Indeks Vektor](#membangun-ulang-database--indeks-vektor)
   - [Menjalankan Aplikasi](#menjalankan-aplikasi)
7. [Deployment di Streamlit Community Cloud](#-deployment-di-streamlit-community-cloud)
8. [Menjalankan dengan Docker](#-menjalankan-dengan-docker)
9. [Pengujian Otomatis & Verifikasi Mutu](#-pengujian-otomatis--verifikasi-mutu)
10. [Basis Data & Asal Data Pangan](#-basis-data--asal-data-pangan)
11. [Kebijakan Keamanan & Privasi](#-kebijakan-keamanan--privasi)
12. [Lisensi & Pengembang](#-lisensi--pengembang)

---

## 🌿 Ikhtisar Proyek (Overview)

**NutriChatBot** adalah platform asisten nutrisi berbasis kecerdasan buatan yang dikembangkan khusus untuk menjawab tantangan gizi masyarakat Indonesia. Alih-alih mengandalkan model bahasa generik (*Large Language Model*) yang kerap mengalami halusinasi dan minim konteks masakan nusantara, NutriChatBot menggabungkan:

- **1.346 Komposisi Pangan Terverifikasi**: Merujuk pada standar resmi Daftar Komposisi Bahan Makanan (DKBM) Kementerian Kesehatan RI per 100 gram bagian yang dapat dimakan (BDD).
- **Dual-Engine Hybrid Retrieval ($wRRF$)**: Memadukan pencarian kata kunci eksak SQLite dengan pencarian semantik vektor ChromaDB (`text-embedding-3-small`).
- **Inferensi Model Penalaran GLM-5.3-flash**: Menggunakan gateway SumoPod AI via OpenAI-compatible SDK dengan kemampuan streaming respon kata demi kata (*token-by-token*).
- **Kalkulator Kalori & Makronutrisi Standar Medis**: Mengimplementasikan formula Mifflin-St Jeor yang tervalidasi klinis oleh *American Dietetic Association* (ADA).
- **Antarmuka Ramah Pengguna Awam (User-Centered Design)**: Dilengkapi tombol pintasan pertanyaan cepat (*quick prompt chips*), kartu ringkasan visual pemenang gizi, lencana status BMI, dan konsep proporsi "Isi Piringku".

---

## ✨ Fitur Unggulan (Key Features)

### 1. 💬 Tanya Gizi (Conversational Hybrid RAG)
- **Pencarian Cerdas Dua Jalur**: Menemukan bahan makanan lokal yang spesifik (*misal: beras giling vs beras ketan*) sekaligus memahami pertanyaan konseptual (*misal: "camilan tinggi serat rendah gula"*).
- **Respon Streaming Cepat**: Jawaban di-generate secara real-time via `st.write_stream` untuk meminimalkan waktu tunggu (*Low Time-to-First-Token*).
- **Tombol Pintasan Pertanyaan Cepat**: Menghilangkan kebingungan pengguna awam (*blank-canvas paralysis*) dengan chip sekali klik:
  - 🍗 *Kalori Dada Ayam vs Paha*
  - 🍳 *Gizi Sepiring Nasi Goreng*
  - 🥑 *Lemak Sehat Alpukat*
  - 🥗 *Camilan Sehat Rendah Gula*
- **Transparansi Rujukan Gizi**: Setiap jawaban menyertakan rujukan komposisi nutrisi di dalam expander `📚 Rujukan Data Nutrisi`.

### 2. 🍳 Masak Apa Hari Ini? (Generator Resep Sehat)
- **Pemanfaatan Bahan Dapur**: Rekomendasi resep masakan bernutrisi lengkap dengan estimasi kalori dan makronutrisi dari bahan yang tersedia di kulkas.
- **Preset Pilihan Bahan Cepat**: Tombol praktis *Ayam + Brokoli*, *Tahu + Tempe + Bayam*, atau *Ikan + Tomat*.
- **Filter Kondisi Kesehatan & Diet Khusus**:
  - *Biasa (Gizi Seimbang)*
  - *Tinggi Protein (Pembangun Otot & Kenyang Awet)*
  - *Rendah Garam (Khusus Penderita Hipertensi)*
  - *Rendah Gula (Khusus Penderita Diabetes)*
  - *Vegetarian & Vegan*
  - *Bebas Gluten (Ramah Sensitivitas Lambung/Celiac)*
- **Unduh Resep**: Tombol unduh resep dalam format Markdown (`.md`).

### 3. 📅 Perencana Menu Mingguan & Kalkulator Gizi Personal
- **Perencana 7 Hari (Senin - Minggu)**: Rencana makan seimbang yang otomatis disesuaikan dengan target kalori harian.
- **Kalkulator BMR & TDEE Terintegrasi**: Menggunakan rumus Mifflin-St Jeor berdasarkan usia, jenis kelamin, berat, tinggi, dan tingkat aktivitas fisik.
- **Visualisasi Hasil Ramah Pemula**:
  - Lencana status BMI berwarna (`🟢 Ideal` vs `🟡 Perlu Perhatian`).
  - Pembagian porsi seimbang konsep **Isi Piringku**: Protein 25%, Lemak Sehat 25%, dan Karbohidrat Kompleks 50%.
- **Unduh Menu**: Simpan rencana menu mingguan ke dalam format Markdown (`.md`).

### 4. 📊 Bandingkan Makanan (Visual Nutrient Comparison)
- **Komparasi Berdampingan**: Menampilkan data kalori, protein, lemak, dan karbohidrat antara dua makanan secara instan.
- **Contoh Preset Populer**: Tombol sekali klik untuk *Tempe vs Tahu*, *Beras Putih vs Merah*, *Dada Ayam vs Sapi*, *Telur vs Ikan*.
- **Kartu Kesimpulan Cepat (*Quick Takeaway*)**: Penanda otomatis siapa yang **Lebih Rendah Kalori** (penghematan kkal) dan siapa yang **Lebih Tinggi Protein** (keunggulan gram).
- **Grafik Interaktif Altair**: Visualisasi grafik batang untuk perbandingan makronutrisi.

### 5. 🥗 Alternatif Sehat
- **Pencari Substitusi Bahan**: Menemukan bahan pengganti yang lebih padat gizi untuk bahan tinggi kalori atau rendah nutrisi.
- **Chip Bahan Favorit**: Pilihan cepat untuk bahan yang sering ingin diganti: *Gula Pasir*, *Minyak Kelapa Sawit*, *Santan Kental*, dan *Tepung Terigu*.

### 6. 🛡️ Guardrail Keselamatan Medis (Clinical Safety)
- **Deteksi Komorbiditas Otomatis**: Memindai kata kunci risiko (diabetes, hipertensi, ginjal, asam urat) pada kueri dan menyuntikkan instruksi kehati-hatian klinis pada prompt AI.
- **Disclaimer Medis Wajib**: Setiap rekomendasi diet dan resep secara otomatis menyertakan peringatan edukatif resmi agar pengguna berkonsultasi dengan dokter gizi klinis (Sp.GK).

---

## 🧠 Arsitektur & Landasan Rekayasa

NutriChatBot dibangun dengan arsitektur 5 lapisan hierarkis (*5-tier architecture*):

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           1. PRESENTATION LAYER (UI/UX)                         │
│   Streamlit Web Interface (Desktop 1600x1000 & Mobile 390x844)                  │
│   - Custom CSS Design Tokens, Rounded Metric Cards & Chip Button Pills          │
│   - User-Centered Design (Bebas Blank-Canvas, Status Asisten Ramah)             │
│   - Collapsible Developer Telemetry Drawer (Pemisahan Observabilitas Teknis)   │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Input Kueri & Biometrik
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      2. GUARDRAILS & PREPROCESSING ENGINE                       │
│   - Pembersihan Stopword Bahasa Indonesia & Ekstraksi Entitas Pangan             │
│   - Detektor Risiko Komorbiditas (Diabetes, Ginjal, Hipertensi, Asam Urat)      │
│   - Validasi Batas Fisik & Sanitasi Parameter Kueri                            │
│   - SensitiveDataFilter (Penyensoran Token API Otomatis pada Logging)           │
└───────────────────┬─────────────────────────────────────────┬───────────────────┘
                    │ Token Pangan Eksak                      │ Vektor Semantik
                    ▼                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     3. DUAL-ENGINE HYBRID RETRIEVAL & FUSION                    │
│   ┌───────────────────────────────────┐ ┌───────────────────────────────────┐   │
│   │     Pencarian Leksikal SQLite     │ │    Pencarian Semantik ChromaDB    │   │
│   │     Kueri Berparameter sub-ms     │ │    text-embedding-3-small (1536d) │   │
│   │     Pencocokan Kata Kunci Pangan  │ │    Koleksi Vektor Cosine Distance │   │
│   └─────────────────┬─────────────────┘ └─────────────────┬─────────────────┘   │
│                     └───────────────────┬─────────────────┘                     │
│                                         ▼                                       │
│                Weighted Reciprocal Rank Fusion (wRRF, k=60)                     │
│                Score(d) = [0.6 / (60 + r_dense)] + [0.4 / (60 + r_lexical)]     │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │ Top-K Konteks Gizi + Prompt Klinis
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          4. LLM INFERENCE & STREAMING                           │
│   - Gateway: SumoPod AI (https://ai.sumopod.com/v1)                             │
│   - Model: GLM-5.3-flash (OpenAI-compatible Python SDK)                         │
│   - Generator Streaming Real-Time (st.write_stream)                             │
│   - Penambahan Disclaimer Medis Wajib pada Output                               │
└────────────────────────────────────────┬────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        5. PERSISTENT STORAGE & DATASETS                         │
│   - SQLite (nutrition.db): 1.346 Baris Komposisi Pangan Terstruktur (DKBM)      │
│   - ChromaDB (/chroma): Penyimpanan Vektor Persisten HNSW/DuckDB               │
│   - CSV Induk (data/nutrition.csv): Standar 100g Porsi Dapat Dimakan            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Dual-Engine Hybrid Retrieval ($wRRF$)
Pencarian vektor murni sering mengalami kegagalan pada nama bahan pangan lokal yang spesifik (*misal membedakan beras merah dan beras putih*), sedangkan pencarian kata kunci murni gagal pada pertanyaan konseptual. NutriChatBot menggabungkan keduanya menggunakan algoritma **Weighted Reciprocal Rank Fusion**:

$$wRRF\_Score(d) = \frac{w_{dense}}{k + rank_{dense}(d)} + \frac{w_{lexical}}{k + rank_{lexical}(d)}$$

- $k = 60$ (konstanta perata *smoothing*)
- $w_{dense} = 0.6$ (bobot semantik)
- $w_{lexical} = 0.4$ (bobot eksak kata kunci)

### 2. Kalkulator Metabolik Klinis (Mifflin-St Jeor)
Menggantikan rumus usang Harris-Benedict (1919) yang kerap melebih-lebihkan kalori hingga 15%. Formula Mifflin-St Jeor direkomendasikan oleh ADA dengan akurasi deviasi di bawah 10%:

$$\text{BMR}_{\text{pria}} = 10 \times \text{berat(kg)} + 6.25 \times \text{tinggi(cm)} - 5 \times \text{usia(th)} + 5$$

$$\text{BMR}_{\text{wanita}} = 10 \times \text{berat(kg)} + 6.25 \times \text{tinggi(cm)} - 5 \times \text{usia(th)} - 161$$

$$\text{TDEE} = \text{BMR} \times \text{Faktor Aktivitas}$$

### 3. Guardrail Keselamatan Medis & Redaksi Data
- **Deteksi Risiko Klinis**: Otomatis mendeteksi kondisi patologis pengguna dan menyuntikkan peringatan klinis pada prompt instruksi AI.
- **SensitiveDataFilter**: Menggunakan regex `(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,})` pada logging handler root. Setiap token API yang tercetak otomatis disensor menjadi `[REDACTED_API_KEY]`.
- **Kueri Berparameter**: Menghilangkan 100% risiko *SQL Injection* dengan menggunakan binding parameter SQLite (`?`).

### 4. User-Centered Design (UCD) untuk Pengguna Awam
- **Pemisahan Mode Konsumen vs Developer**: Menghilangkan jargon developer (nama model LLM, dimensi vektor, endpoint) dari pandangan utama dan memindahkannya ke dalam *drawer expander*.
- **State-Hydrated Prompt Chips**: Tombol cepat yang secara otomatis mengisi state input tanpa mengharuskan pengguna mengetik di keyboard ponsel/desktop.
- **Kesimpulan Visual Cepat**: Kalkulasi otomatis selisih gizi antar makanan yang disajikan dalam bentuk poin sorotan langsung.

---

## 🗂️ Struktur Direktori Proyek

```text
NutriChatBot/
├── app.py                      # Titik masuk aplikasi Streamlit (UI, Styling, Chat & Visualisasi)
├── config.py                   # Konfigurasi terpusat, dataclass AppSettings & filter redaksi log
├── embeddings.py               # Facade kompatibilitas fungsi RAG & embedding
├── services/                   # Layanan bisnis modular
│   ├── database.py             # Manajemen SQLite, sanitasi parameter, validasi CSV & keyword ranking
│   ├── embeddings.py           # ChromaDB client, indexing batch & Hybrid RRF fusion
│   ├── llm.py                  # Inisialisasi klien SumoPod AI, streaming generator & guardrail klinis
│   └── nutrition_calc.py       # Kalkulasi biometrik metabolik (BMR, TDEE, BMI, Macro Split)
├── scripts/
│   └── rebuild_index.py        # CLI tool untuk inisialisasi & rebuild SQLite & ChromaDB vektor
├── tests/                      # Rangkaian pengujian otomatis pytest (51 tests)
│   ├── conftest.py             # Fixtures & mock client
│   ├── test_app_ui.py          # Pengujian integrasi antarmuka Streamlit via AppTest
│   ├── test_config.py          # Pengujian konfigurasi & filter penyensoran rahasia
│   ├── test_data_integrity.py  # Pengujian integritas berkas CSV & dataset 1.346 pangan
│   ├── test_database.py        # Pengujian database SQLite, SQLi & deteksi anomali
│   ├── test_embeddings.py      # Pengujian ChromaDB, sanitasi NaN & algoritma wRRF
│   ├── test_llm.py             # Pengujian inferensi streaming, guardrails & error handling
│   └── test_nutrition_calc.py  # Pengujian formula Mifflin-St Jeor & pembagian makro
├── data/
│   └── nutrition.csv           # Dataset gizi pangan Indonesia (1.346 record, standar 100g BDD)
├── portfolio-export/           # Paket ekspor studi kasus profesional bilingual & aset visual
├── Dockerfile                  # Multi-stage production container build
├── .dockerignore               # Aturan pengabaian berkas Docker
├── .github/workflows/ci.yml    # Pipeline otomatis GitHub Actions CI (Linter & Pytest)
├── .env.example                # Template variabel lingkungan
├── pyproject.toml              # Definisi dependensi & konfigurasi tooling (Ruff & Pytest)
└── uv.lock                     # Lockfile resolusi dependensi deterministik
```

---

## 💻 Tech Stack

| Komponen | Teknologi | Keterangan |
| :--- | :--- | :--- |
| **Bahasa Utama** | Python 3.10 / 3.11 / 3.12 | Strongly typed, modular service architecture |
| **Antarmuka (Frontend)** | Streamlit | Responsive multi-tab UI dengan Custom CSS tokens |
| **Model Bahasa (LLM)** | GLM-5.3-flash | Reasoning model via SumoPod AI Gateway |
| **Model Embedding** | text-embedding-3-small | Vektor 1536-dimensi via SumoPod AI Gateway |
| **Database Relasional** | SQLite3 | Penyimpanan kueri leksikal lokal & komparasi tabular |
| **Vector Database** | ChromaDB | Penyimpanan persisten vektor dokumen & pencarian kosinus |
| **Visualisasi Data** | Altair / Vega-Lite | Grafik batang perbandingan makronutrisi interaktif |
| **Package Manager** | uv (Astral) | Resolusi dependensi super cepat dan deterministik |
| **Pengujian (Testing)** | Pytest & Streamlit AppTest | 51 unit, integration, dan headless UI test cases |
| **Linter & Formatter** | Ruff | Pemeriksaan mutu kode cepat berstandar PEP 8 |
| **Kontainerisasi** | Docker | Multi-stage build dengan user non-root aman |

---

## 🚀 Panduan Instalasi & Menjalankan

### Prasyarat

- Python 3.10+ (Direkomendasikan Python 3.12)
- [uv](https://astral.sh/uv) (Alat manajemen paket modern berbasis Rust)
- API Key SumoPod AI aktif ([Daftar di SumoPod AI](https://ai.sumopod.com/))

### Langkah Setup dengan `uv` (Direkomendasikan)

```bash
# 1. Clone repository
git clone https://github.com/victoriavals/NutriChatBot.git
cd NutriChatBot

# 2. Buat virtual environment dan install dependensi (termasuk dev tools)
uv venv
uv sync --extra dev

# 3. Siapkan file konfigurasi lingkungan
cp .env.example .env
```

> **Alternatif menggunakan pip tradisional:**
> ```bash
> python -m venv .venv
> # Windows:
> .venv\Scripts\activate
> # Linux/macOS:
> source .venv/bin/activate
> pip install -e ".[dev]"
> ```

---

### Konfigurasi Environment (`.env`)

Buka file `.env` dan masukkan API key SumoPod AI Anda:

```env
# WAJIB: API Key dari SumoPod AI (format: sk-...)
API_KEY=your_sumopod_api_key_here

# OPSIONAL: Endpoint URL gateway (default: https://ai.sumopod.com/v1)
BASE_URL=https://ai.sumopod.com/v1

# OPSIONAL: Model penalaran LLM (default: glm-5.3-flash)
MODEL_LLM=glm-5.3-flash

# OPSIONAL: Model embedding vektor (default: text-embedding-3-small)
MODEL_EMBEDDING=text-embedding-3-small
```

---

### Membangun Ulang Database & Indeks Vektor

Aplikasi secara otomatis menginisialisasi database SQLite `nutrition.db` saat pertama kali dijalankan. Untuk membangun ulang indeks vektor ChromaDB:

```bash
# Opsi 1: Bangun ulang SQLite dan ChromaDB sekaligus (memanggil Embeddings API)
uv run python scripts/rebuild_index.py --clean

# Opsi 2: Hanya membangun ulang SQLite tanpa memanggil Embeddings API (Cepat & Hemat)
uv run python scripts/rebuild_index.py --skip-chroma
```

*Catatan: Anda juga dapat membangun ulang indeks secara langsung melalui menu **⚙️ Index Data** di dalam aplikasi.*

---

### Menjalankan Aplikasi

Jalankan server aplikasi Streamlit lokal:

```bash
uv run streamlit run app.py
```

Aplikasi akan otomatis terbuka di peramban web Anda pada alamat:  
👉 **`http://localhost:8501`**

---

## 🚀 Deployment di Streamlit Community Cloud

NutriChatBot telah dipublikasikan secara resmi di **Streamlit Community Cloud** dan dapat diakses publik langsung dari peramban web:

🌐 **Akses Demo Publik**: **[https://nutrichatbot.streamlit.app/](https://nutrichatbot.streamlit.app/)**

### Mekanisme & Karakteristik Deployment:
1. **Continuous Deployment dari GitHub**: Repositori Streamlit Cloud terintegrasi dengan branch `main` GitHub (`victoriavals/NutriChatBot`). Setiap `git push` ke branch `main` akan memicu kompilasi container dan pembaruan aplikasi secara otomatis.
2. **Pengaturan Secrets Produksi**: Sesuai prinsip keamanan data, file `.env` tidak pernah dikomit ke repositori. Kredensial produksi diatur melalui menu **Settings $\rightarrow$ Secrets** pada konsol [Streamlit Cloud](https://share.streamlit.io):
   ```toml
   API_KEY = "your_production_api_key"
   BASE_URL = "https://ai.sumopod.com/v1"
   MODEL_LLM = "glm-5.3-flash"
   MODEL_EMBEDDING = "text-embedding-3-small"
   ```
3. **Penanganan Auto-Sleep**: Pada tier komunitas gratis Streamlit, aplikasi yang tidak menerima lalu lintas selama beberapa hari akan beralih ke mode tidur (*asleep*). Pengunjung cukup menekan tombol **"Yes, get this app back up!"** untuk mengaktifkan kembali server dalam 1–2 menit.

---

## 🐳 Menjalankan dengan Docker

NutriChatBot dilengkapi `Dockerfile` multi-stage berbasis Debian Slim yang aman dan ringan:

```bash
# 1. Build image Docker
docker build -t nutrichatbot:latest .

# 2. Jalankan container dengan file .env
docker run -d \
  -p 8501:8501 \
  --env-file .env \
  --name nutrichatbot-container \
  nutrichatbot:latest
```

Buka peramban di `http://localhost:8501`.

---

## 🧪 Pengujian Otomatis & Verifikasi Mutu

Seluruh fungsi aplikasi diverifikasi secara ketat melalui piramida pengujian otomatis:

```bash
# Jalankan seluruh 51 automated tests (unit, integrasi, dan headless UI)
uv run pytest -v

# Jalankan linter dan format check menggunakan Ruff
uv run ruff check .

# Jalankan kompilasi sintaks Python
uv run python -m py_compile app.py config.py embeddings.py services/*.py scripts/*.py
```

### Ringkasan Cakupan Uji (51 Passed):
- `tests/test_nutrition_calc.py` (10 tests): Validasi formula Mifflin-St Jeor, batas numerik biometrik, rasio makro kalori.
- `tests/test_database.py` (9 tests): Sanitasi parameter SQLi, deteksi anomali gizi, ekstraksi stopword.
- `tests/test_embeddings.py` (8 tests): Sanitasi nilai NaN, batching ChromaDB, algoritma fusi wRRF.
- `tests/test_llm.py` (15 tests): Streaming generator, proteksi token rahasia, guardrail komorbiditas klinis.
- `tests/test_app_ui.py` (3 tests): Simulasi headless siklus hidup antarmuka Streamlit via `AppTest`.
- `tests/test_config.py` (4 tests): Pembacaan `.env` dan filter sensor logging.
- `tests/test_data_integrity.py` (2 tests): Integritas struktur 1.346 record CSV pangan Indonesia.

---

## 📊 Basis Data & Asal Data Pangan

- **Total Record**: 1.346 bahan makanan dan hidangan khas Indonesia.
- **Standar Porsi**: 100 gram Bagian yang Dapat Dimakan (BDD / *Edible Portion*).
- **Zat Gizi yang Tercakup**: Energi (Kalori), Protein, Lemak, Karbohidrat, Serat, Kalsium, Fosfor, Besi, Natrium, Kalium, Tembaga, Seng, Retinol, Beta-Karoten, Vitamin B1, B2, Niasin, dan Vitamin C.
- **Rujukan Data**: Diadaptasi dari tabel resmi **Daftar Komposisi Bahan Makanan (DKBM) Kementerian Kesehatan RI** via dataset publik Kaggle oleh [Anas Fikri Hanif](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset).

---

## 🛡️ Kebijakan Keamanan & Privasi

1. **Nir-Bocor Kredensial (*Zero Secret Leakage*)**: Berkas `.env` dimasukkan ke dalam `.gitignore` dan `.dockerignore`. Logger aplikasi mengimplementasikan sensor otomatis untuk pola API key.
2. **Kueri Kebal SQL Injection**: Kueri pencarian SQLite menerapkan binding parameter dan pelolosan karakter wildcard (`%` dan `_`).
3. **Batasan Disclaimer Medis**: NutriChatBot dirancang sebagai media edukasi dan panduan hidup sehat, bukan alat diagnosa medis pengganti konsultasi dokter spesialis gizi klinis.

---

## 📜 Lisensi & Pengembang

- **Lisensi**: [MIT License](LICENSE) — Bebas digunakan, dipelajari, dan dikembangkan untuk keperluan edukasi dan riset.
- **Pengembang & Arsitek**: **Naufal Firdaus** ([@victoriavals](https://github.com/victoriavals))
- **Email**: `naufalfirdaus282@gmail.com`
