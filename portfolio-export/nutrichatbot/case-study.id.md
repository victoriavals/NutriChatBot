# Studi Kasus: NutriChatBot — Asisten Nutrisi & Pola Makan Cerdas dengan Hybrid RAG

**Penulis:** Naufal Firdaus  
**Repository:** [https://github.com/victoriavals/NutriChatBot](https://github.com/victoriavals/NutriChatBot)  
**Teknologi Utama:** Python 3.12, Streamlit, ChromaDB, SQLite, OpenAI SDK / SumoPod AI, GLM-5.3-flash, Docker  
**Tanggal:** 15 November 2026 (Diperbarui dengan Arsitektur UI/UX Berpusat pada Pengguna)  

---

## 1. Ikhtisar Proyek (Overview)

NutriChatBot adalah aplikasi asisten nutrisi dan perencana pola makan cerdas yang dirancang khusus untuk memberikan panduan gizi akurat dan relevan dengan kebiasaan pangan masyarakat Indonesia. Alih-alih mengandalkan model bahasa generik tanpa landasan fakta, NutriChatBot mengintegrasikan arsitektur **Dual-Engine Hybrid Retrieval-Augmented Generation (Hybrid RAG)** yang didukung oleh 1.346 data komposisi pangan terverifikasi (standar per 100 gram porsi dapat dimakan dari rujukan DKBM / Kementerian Kesehatan RI).

Yang membedakan, proyek ini memadukan ketangguhan rekayasa *retrieval* algoritma dengan pendekatan **User-Centered Design (UCD)** yang dirancang khusus bagi **pengguna awam** (ibu rumah tangga, mahasiswa, pemula diet, dan orang tua). Dengan menghapus istilah teknis yang membingungkan, mengatasi kebingungan saat pertama kali mengetik (*blank-canvas paralysis*) melalui tombol *chip* pertanyaan cepat, dan menerjemahkan angka metabolik menjadi kesimpulan gizi visual yang praktis, NutriChatBot menjembatani teknologi AI mutakhir dengan kemudahan pakai harian.

Fitur utama yang diimplementasikan meliputi:
- **Tanya Gizi:** Tanya-jawab gizi interaktif berbasis percakapan dengan streaming token real-time, tombol *chip* pertanyaan cepat, serta ekspander sitasi rujukan nutrisi.
- **Perencana Menu Mingguan:** Rencana menu 7 hari terpersonalisasi yang dihitung secara presisi menggunakan formula metabolik standar klinis Mifflin-St Jeor, lengkap dengan lencana status BMI dan visualisasi porsi makronutrisi "Isi Piringku".
- **Bandingkan Makanan:** Analisis komparasi makronutrisi berdampingan yang divisualisasikan dengan grafik interaktif Altair dan kartu kesimpulan cepat (*Quick Takeaway*) otomatis.
- **Masak Apa Hari Ini?:** Generator resep sehat berbasis bahan dapur yang dimiliki dengan tombol preset kombinasi bahan dan filter batasan penyakit (rendah garam untuk hipertensi, rendah gula untuk diabetes).
- **Alternatif Sehat:** Pencari substitusi bahan makanan yang lebih padat gizi dan rendah kalori disertai tombol pilihan bahan populer.

---

## 2. Masalah yang Ditangani (Problem Statement)

Akses masyarakat terhadap informasi nutrisi yang akurat dan praktis di Indonesia terhambat oleh empat masalah mendasar:

1. **Halusinasi pada LLM Komersial Generik:** Model bahasa umum kerap mengarang takaran kalori dan makronutrisi saat ditanya mengenai masakan lokal Indonesia. Model sering gagal membedakan dampak metode pengolahan (misalnya tempe goreng menyerap minyak sehingga kalorinya melonjak hingga 336 kkal/100g, dibandingkan tahu goreng sebesar 115 kkal/100g).
2. **Kesenjangan Pencarian Pangan (Vocabulary Mismatch):** Pencarian semantik murni (dense vector) sering kesulitan membedakan taksonomi pangan yang mirip (*Beras giling merah* vs *Beras ketan hitam*), sedangkan pencarian kata kunci murni gagal menangkap pertanyaan konseptual ("makanan berprotein tinggi yang murah dan mudah didapat").
3. **Ketiadaan Batasan Keselamatan Medis (Clinical Guardrails):** Sebagian besar chatbot AI kesehatan memberikan saran medis tanpa mendeteksi komorbiditas kritis (diabetes, gagal ginjal, hipertensi) dan tidak menyertakan disclaimer medis wajib.
4. **"Blank-Canvas Paralysis" & Hambatan Kognitif Pengguna Awam:** Pengguna awam yang tidak terbiasa dengan *prompt engineering* sering bingung ketika dihadapkan pada kolom teks kosong dan istilah rumit (seperti menentukan target kalori tanpa paham BMR vs TDEE). Selain itu, metrik developer (nama model, dimensi vektor, endpoint) sering mengaburkan fokus pengguna awam.

---

## 3. Kontribusi Pribadi (Personal Contribution)

Sebagai perancang dan pengembang utama (author), Naufal Firdaus memimpin dan mengeksekusi seluruh siklus teknis proyek ini:

- **Arsitektur Hybrid Retrieval:** Merancang dan mengimplementasikan algoritma Weighted Reciprocal Rank Fusion ($wRRF$) pada `services/embeddings.py`, menyatukan kemiripan kosinus dari ChromaDB (`text-embedding-3-small`) dengan pencarian leksikal berperingkat dari SQLite.
- **Mesin Perhitungan Metabolik:** Membangun modul `services/nutrition_calc.py` yang mengimplementasikan rumus Basal Metabolic Rate (BMR) dan Total Daily Energy Expenditure (TDEE) Mifflin-St Jeor, lengkap dengan validasi batas fisik dan pembagian rasio kalori makronutrisi.
- **Rekayasa UI/UX Berpusat pada Pengguna Awam:** Membangun sistem desain ramah konsumen pada `app.py`:
  - Menyuntikkan CSS kustom untuk tata letak modern, kartu metrik lembut, dan tombol *pill-chips* ergonomis.
  - Membangun mekanisme *interactive prompt chips* di seluruh menu untuk menghapus kebingungan pengguna saat pertama kali membuka aplikasi.
  - Memisahkan telemetri diagnostik pengembang ke dalam expander terpisah, mengutamakan status asisten yang hangat dan menenangkan bagi pengguna biasa.
  - Mengimplementasikan logika pembanding otomatis untuk memberikan kesimpulan gizi verbal seketika tanpa menuntut pengguna berhitung manual.
- **Guardrail & Keamanan Data:** Mengembangkan deteksi risiko klinis pada `services/llm.py` untuk mengidentifikasi kata kunci komorbiditas berisiko tinggi (diabetes, ginjal, hipertensi, asam urat) dan menyuntikkan peringatan klinis secara otomatis. Mengimplementasikan `SensitiveDataFilter` pada `config.py` guna meredaksi API key pada logging.
- **Pengujian & Kesiapan Produksi:** Mendirikan piramida pengujian otomatis dengan 51 uji (unit, integrasi, UI) berstatus 100% lulus, didukung kontainerisasi Docker multi-stage dan GitHub Actions CI.

---

## 4. Fokus Rekayasa (Engineering Focus)

Fokus utama rekayasa perangkat lunak pada proyek ini mencakup lima pilar penting:

### A. Dual-Engine Hybrid Retrieval dengan wRRF
Untuk memecahkan kelemahan retrieval satu paradigma, NutriChatBot menerapkan pipeline pencarian ganda:
- **Jalur Leksikal:** Mengeksekusi pencarian teks berindeks di SQLite (`search_nutrition_keyword_ranked`), memfilter stopword percakapan bahasa Indonesia (*yang*, *dan*, *berapa*, *apakah*), dan menghitung skor relevansi berdasarkan kecocokan nama dan kategori pangan.
- **Jalur Semantik Dense:** Menghasilkan vektor 1536-dimensi via `text-embedding-3-small` yang disimpan dalam ChromaDB dengan metrik jarak kosinus.
- **Lapisan Fusi (wRRF):** Menggabungkan urutan peringkat menggunakan rumus Weighted Reciprocal Rank Fusion:
  $$wRRF\_Score(d) = \frac{w_{dense}}{k + rank_{dense}(d)} + \frac{w_{lexical}}{k + rank_{lexical}(d)}$$
  dengan parameter $k=60$, $w_{dense}=0.6$, dan $w_{lexical}=0.4$. Hasilnya, istilah bahan pangan persis akan menempati urutan teratas, sementara item yang relevan secara semantik memperkaya konteks.

### B. Rekayasa UX untuk Pengguna Awam (Human-Computer Interaction)
Menerapkan teknik interaksi khusus untuk memudahkan pemula:
- **State-Hydrated Prompt Chips:** Mengintegrasikan tombol pintasan pertanyaan populer yang secara otomatis mengisi state input tanpa mengetik manual.
- **Kesimpulan Visual Otomatis (Takeaway Cards):** Menghitung selisih nilai makro ($|Kal_1 - Kal_2|$, $|Prot_1 - Prot_2|$) dan menyajikannya dalam kartu sorotan sederhana (opsi mana yang lebih rendah kalori dan mana yang lebih kaya protein).
- **Edukasi Kalori Bersahabat:** Menyajikan panduan konsep "Isi Piringku" (25% Protein, 25% Lemak Sehat, 50% Karbohidrat Kompleks) yang didampingi lencana warna status BMI (`🟢 Ideal` vs `🟡 Perlu Perhatian`).

### C. Inferensi Streaming Berlatensi Rendah
Mengintegrasikan gateway SumoPod AI berbasis protokol OpenAI dengan model `glm-5.3-flash`. Respons di-streaming ke pengguna potongan demi potongan (*token-by-token*) melalui generator Python, secara drastis memangkas waktu tunggu respon awal (Time-to-First-Token) pada Streamlit.

### D. Proteksi Kredensial Nir-Bocor
Mengimplementasikan subclass kustom `logging.Filter` (`SensitiveDataFilter`) yang memindai pesan log dan argumen fungsi terhadap ekspresi reguler `(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,})`. Setiap token API yang terdeteksi langsung digantikan dengan `[REDACTED_API_KEY]`, mencegah kebocoran pada log kontainer maupun stdout.

### E. Piramida Pengujian Otomatis
Membangun 51 skenario pengujian otomatis yang mencakup:
- Batasan biometrik fisik dan validitas perhitungan matematika.
- Ketahanan terhadap SQL injection menggunakan kueri berparameter.
- Idempotensi database dan deteksi anomali energi makronutrisi.
- Sanitasi metadata embedding (penanganan nilai `NaN`, `Inf`, dan tipe numpy).
- Transisi state UI dan alur penanganan error menggunakan Streamlit `AppTest`.

---

## 5. Arsitektur Sistem (Architecture)

Arsitektur sistem NutriChatBot terbagi ke dalam 5 lapisan hierarkis:

```
+-----------------------------------------------------------------------------------+
|                            1. LAPISAN PRESENTASI (UI/UX)                          |
|         Aplikasi Web Streamlit Multi-Tab (Desktop 1600x1000 & Seluler 390x844)    |
|  - Token Desain CSS Kustom & Kartu Metrik Lembut                                  |
|  - Chip Pintasan Pertanyaan & Resep Sekali Klik (Bebas Hambatan Mengetik)         |
|  - Pemisahan Drawer Telemetri Pengembang dari Antarmuka Konsumen Biasa            |
|  [Tanya Gizi]   [Masak Apa Hari Ini]   [Perencana Menu]   [Bandingkan Makanan]     |
+------------------------------------------+----------------------------------------+
                                           | Kueri Pengguna & Data Biometrik
                                           v
+-----------------------------------------------------------------------------------+
|                      2. ENGINE GUARDRAIL & PRA-PEMROSESAN                         |
|  - Penghapus Stopword Indonesia & Tokenizer Entitas Pangan                        |
|  - Detektor Risiko Klinis & Komorbiditas (Diabetes, Hipertensi, Ginjal)           |
|  - Validasi Panjang Input & Batasan Nilai Numerik                                 |
|  - SensitiveDataFilter (Redaksi API Key Otomatis)                                 |
+---------------------+-------------------------------------+-----------------------+
                      | Token Pangan Eksak                  | Intensi Semantik
                      v                                     v
+-----------------------------------------------------------------------------------+
|                      3. HYBRID RETRIEVAL DUAL-ENGINE                              |
|  +------------------------------------+   +------------------------------------+  |
|  |     Pencarian Leksikal SQLite      |   |     Pencarian Semantik ChromaDB    |  |
|  |     Database `nutrition.db`        |   |     Koleksi Vektor (/chroma)       |  |
|  |     Pencocokan kata kunci sub-ms   |   |     text-embedding-3-small (1536d) |  |
|  +-----------------+------------------+   +------------------+-----------------+  |
|                    \                                         /                    |
|                     v                                       v                     |
|           +-------------------------------------------------------+               |
|           |       Weighted Reciprocal Rank Fusion (wRRF)          |               |
|           |       Score(d) = w_d/(k + r_d) + w_l/(k + r_l)        |               |
|           +---------------------------+---------------------------+               |
+---------------------------------------|-------------------------------------------+
                                        | Konteks Terformat & System Prompt
                                        v
+-----------------------------------------------------------------------------------+
|                            4. LAPISAN INFERENSI LLM                               |
|  - Gateway: SumoPod AI (https://ai.sumopod.com/v1)                                |
|  - Model: GLM-5.3-flash (Klien kompatibel OpenAI SDK)                             |
|  - Output: Streaming token real-time (`st.write_stream`)                          |
|  - Keselamatan: Penambahan Disclaimer Medis wajib pada seluruh saran gizi         |
+---------------------------------------+-------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                         5. PENYIMPANAN DATA PERSISTEN                             |
|  - SQLite (nutrition.db): 1.346 catatan gizi pangan terstruktur                   |
|  - ChromaDB (/chroma): Indeks vektor DuckDB/Parquet dengan metadata lengkap       |
|  - Sumber Mentah (data/nutrition.csv): Standar porsi 100 gram dapat dimakan       |
+-----------------------------------------------------------------------------------+
```

---

## 6. Detail Implementasi (Implementation)

### Struktur Modul Kode
- `config.py`: Pengaturan terpusat, dataclass `AppSettings`, pembacaan file `.env`, konstanta aplikasi, dan filter `SensitiveDataFilter`.
- `services/database.py`: Validasi file CSV, deteksi anomali rumus energi makronutrisi ($|4P + 4C + 9F - Kalori| > 50$), manajemen skema SQLite, serta pencarian leksikal berperingkat.
- `services/embeddings.py`: Inisialisasi fungsi embedding OpenAI/SumoPod, manajemen koneksi ChromaDB, pengindeksan batch dengan sanitasi nilai `NaN`, serta mesin fusi `hybrid_retrieve_nutrition`.
- `services/nutrition_calc.py`: Modul matematika murni untuk menghitung BMR/TDEE Mifflin-St Jeor, faktor pengali aktivitas fisik, status BMI, dan distribusi makronutrisi.
- `services/llm.py`: Peracikan prompt terstruktur, pencegatan risiko klinis, dan generator streaming token untuk fitur tanya-jawab, resep sehat, rencana menu, dan alternatif bahan.
- `app.py`: Titik masuk aplikasi Streamlit yang mengelola styling CSS kustom (`_inject_custom_styles`), hidrasi session state untuk *chips*, diagram Altair, dan navigasi yang ramah pengguna awam.

### Formula Matematika Utama

#### 1. BMR Mifflin-St Jeor:
$$\text{BMR}_{\text{pria}} = 10 \times \text{berat (kg)} + 6.25 \times \text{tinggi (cm)} - 5 \times \text{usia (tahun)} + 5$$
$$\text{BMR}_{\text{wanita}} = 10 \times \text{berat (kg)} + 6.25 \times \text{tinggi (cm)} - 5 \times \text{usia (tahun)} - 161$$

#### 2. Kebutuhan Energi Total (TDEE):
$$\text{TDEE} = \text{BMR} \times \text{Faktor Aktivitas}$$
*(Pengali: Sedentary = 1.2, Ringan = 1.375, Sedang = 1.55, Berat = 1.725)*

#### 3. Distribusi Makronutrisi Target:
- Kalori Target = $\text{TDEE} \pm \Delta_{\text{tujuan}}$
- Protein: $25\%$ kalori ($4\text{ kkal/g}$)
- Lemak: $25\%$ kalori ($9\text{ kkal/g}$)
- Karbohidrat: $50\%$ kalori ($4\text{ kkal/g}$)

---

## 7. Keputusan Teknis & Kompromi (Technical Decisions)

| Keputusan | Alternatif Dipertimbangkan | Alasan Pemilihan |
| :--- | :--- | :--- |
| **User-Centered Design (UCD) dengan Prompt Chips** | Kotak Teks Kosong Statis | Pengguna awam sering mengalami keraguan saat melihat kotak kosong. Tombol pintasan pertanyaan sekali klik memungkinkan eksplorasi langsung tanpa mengetik manual, sedangkan CSS kustom meningkatkan estetika ke tingkat aplikasi komersial. |
| **Pemisahan Telemetri Developer** | Metrik Teknis Ditampilkan Bebas di Sidebar | Menampilkan nama LLM, dimensi vektor, dan endpoint membingungkan pengguna non-teknis. Menyimpannya di dalam drawer tersembunyi mempertahankan observabilitas pengembang tanpa mengorbankan kenyamanan pengguna biasa. |
| **Hybrid wRRF Retrieval** | Pencarian Vektor Murni (ChromaDB saja) | Pencarian vektor murni sering gagal membedakan nama pangan spesifik Indonesia (*Beras merah* vs *Beras putih*). Memadukan pencarian leksikal menjamin presisi 100% pada kata kunci eksak sembari mempertahankan konteks semantik. |
| **SumoPod AI + GLM-5.3-flash** | Google Gemini atau OpenAI GPT-4o langsung | SumoPod AI menyediakan gateway kompatibel OpenAI dengan model penalaran GLM-5.3-flash yang menawarkan efisiensi biaya dan performa inferensi tinggi pada pengujian lokal. |
| **Formula Mifflin-St Jeor** | Harris-Benedict (1919) | American Dietetic Association (ADA) merekomendasikan Mifflin-St Jeor sebagai formula paling akurat secara klinis (deviasi di bawah 10% pada lebih dari 82% subjek uji) dibandingkan Harris-Benedict yang kerap melebih-lebihkan kalori hingga 15%. |
| **SQLite Lokal + ChromaDB** | Database Vektor Cloud (Pinecone / Supabase) | Menghilangkan latensi jaringan eksternal dan biaya sewa cloud untuk dataset 1.346 record, memungkinkan pengembangan offline mandiri dan portabilitas kontainer penuh. |
| **Pengujian Streamlit (`AppTest`)** | Headless Selenium / Playwright | Streamlit `AppTest` memungkinkan simulasi siklus hidup UI secara headless dalam hitungan milidetik di dalam `pytest` tanpa perlu menjalankan peramban berat. |

---

## 8. Bukti Pengujian & Hasil Verifikasi (Evidence and Results)

Implementasi sistem telah diverifikasi secara menyeluruh melalui beberapa lapisan pengujian:

1. **Rangkaian Uji Otomatis Pytest:**
   - **51 kasus uji otomatis** mencakup alur UI, konfigurasi, operasi database, embedding, streaming LLM, dan kalkulator gizi.
   - **Hasil:** 100% LULUS (exit code 0, selesai dalam 5,85 detik).
2. **Uji Regresi Khusus (`scratch/qa_suite_runner.py`):**
   - **18 skenario uji end-to-end** mencakup pencarian entitas eksak, ketahanan SQLi (`OR 1=1`), redaksi API key pada log, injeksi peringatan klinis, dan presisi BMR.
   - **Hasil:** 18/18 LULUS (100% sukses).
3. **Pemeriksaan Kualitas Kode & Linter:**
   - Diperiksa menggunakan **Ruff** pada seluruh repositori: `All checks passed!` (nol warning, kepatuhan penuh PEP 8 dan type hinting).
4. **Verifikasi Visual Antarmuka:**
   - Berhasil mengambil 6 aset visual berkualitas tinggi langsung dari aplikasi Streamlit yang sedang berjalan via Chrome DevTools Protocol (CDP), mengonfirmasi tata letak desktop (1600x1000) dan seluler (390x844) dengan tombol *chip* dan gaya visual terbaru.

---

## 9. Keterbatasan & Batasan Etika (Limitations)

- **Cakupan Pengujian Lokal:** Seluruh pengujian dan tolok ukur dieksekusi pada lingkungan lokal dan alur kontainer Docker. Skalabilitas konkurensi produksi publik multi-pengguna belum terverifikasi secara langsung.
- **Batasan Disclaimer Medis:** NutriChatBot difungsikan secara tegas sebagai alat bantu edukasi nutrisi, bukan perangkat diagnosis medis berlisensi. Setiap saran pola makan wajib menyertakan disclaimer medis yang menganjurkan pengguna dengan kondisi kronis berkonsultasi langsung dengan dokter spesialis gizi klinis (Sp.GK).
- **Cakupan Dataset:** Dataset terbatas pada 1.346 bahan pangan standar Indonesia (DKBM). Makanan olahan pabrikan modern dengan kode batang kemasan berada di luar cakupan database saat ini.
