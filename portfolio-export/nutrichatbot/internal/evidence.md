# Technical Evidence & Verification Matrix: NutriChatBot

**Target Project:** NutriChatBot  
**Commit Inspected:** `22eaaff` (and working tree updates)  
**Author:** Naufal Firdaus (`victoriavals` / `naufalfirdaus282@gmail.com`)  
**Audit Date:** 2026-11-15  

---

## 1. Traceability Matrix: Code Claims to Implementation

| Feature / Claim | Implementation File & Line(s) | Verification Evidence |
| :--- | :--- | :--- |
| **Hybrid Retrieval (wRRF)** | `services/embeddings.py`: Lines 273–345 (`hybrid_retrieve_nutrition`) | `tests/test_embeddings.py::test_hybrid_retrieve_nutrition_merges_and_ranks` (PASS) |
| **Lexical Ranked Search** | `services/database.py`: Lines 220–280 (`search_nutrition_keyword_ranked`) | `tests/test_database.py::test_search_nutrition_keyword_ranked_ranks_multi_term` (PASS) |
| **Stopword Extraction** | `services/database.py`: Lines 190–215 (`extract_food_keywords`, `NUTRITION_STOPWORDS`) | `tests/test_database.py::test_extract_food_keywords_removes_stopwords` (PASS) |
| **Mifflin-St Jeor BMR** | `services/nutrition_calc.py`: Lines 44–80 (`calculate_bmr_mifflin_st_jeor`) | `tests/test_nutrition_calc.py::test_calculate_bmr_mifflin_st_jeor` (PASS) |
| **TDEE & Macro Allocation** | `services/nutrition_calc.py`: Lines 82–170 (`calculate_personalized_nutrition_plan`) | `tests/test_nutrition_calc.py::test_calculate_macro_split_matches_calorie_ratio` (PASS) |
| **Clinical Risk Detection** | `services/llm.py`: Lines 125–155 (`detect_clinical_risk`, `CLINICAL_RISK_KEYWORDS`) | `tests/test_llm.py::test_detect_clinical_risk_identifies_keywords` (PASS) |
| **Streaming Inference** | `services/llm.py`: Lines 180–230 (`answer_nutrition_question_stream`) | `tests/test_llm.py::test_answer_nutrition_question_stream_returns_stream_and_contexts` (PASS) |
| **Sensitive Log Redaction** | `config.py`: Lines 55–69 (`SensitiveDataFilter`, `_API_KEY_PATTERN`) | `tests/test_config.py::test_sensitive_data_filter_redacts_keys` (PASS) |
| **Multi-Tab Streamlit UI** | `app.py`: Lines 95–610 (Tanya Gizi, Resep, Perencana, Bandingkan, Alternatif) | `tests/test_app_ui.py::test_app_all_menus_happy_path_and_indexing` (PASS) |
| **Dataset Integrity** | `data/nutrition.csv`: 1,346 rows, verified schema & columns | `tests/test_data_integrity.py::test_repository_nutrition_csv_integrity` (PASS) |
| **Database Anomaly Checks** | `services/database.py`: Lines 110–145 (`detect_nutrition_anomalies`) | `tests/test_database.py::test_detect_nutrition_anomalies_flags_implausible_macros` (PASS) |

---

## 2. Test Execution Results

### A. Pytest Automated Test Suite
- **Command:** `.venv/Scripts/pytest -v`
- **Total Tests:** 51
- **Passed:** 51
- **Failed:** 0
- **Duration:** 6.18s
- **Exit Code:** 0

### B. Dedicated QA End-to-End Suite (`scratch/qa_suite_runner.py`)
- **Total Checks:** 18
- **Passed:** 18
- **Failed:** 0
- **Summary:**
  - Entity match accuracy: 100% on *Nasi goreng*, *Tempe*, *Tahu*, *Dada ayam*.
  - Parameterized SQL injection defense: Passed (`' OR 1=1 --` produces 0 matches, never escapes parameter binding).
  - Secret redaction: Verified against Google `AIza...` and SumoPod/OpenAI `sk-...` tokens.
  - Math precision: Verified Mifflin-St Jeor values against clinical benchmark calculators.

---

## 3. Visual & UI Capture Verification

All visual assets were captured live via **Chrome DevTools Protocol (CDP)** against a running instance of `app.py`:
- `01-tanya-gizi-chat.webp`: 1600x1000 px, 25,280 bytes.
- `02-kalkulator-bmr-tdee-dan-menu.webp`: 1600x1000 px, 88,738 bytes.
- `03-perbandingan-gizi-visual.webp`: 1600x1000 px, 81,816 bytes.
- `04-generator-resep-sehat.webp`: 1600x1000 px, 94,906 bytes.
- `05-tampilan-mobile-chat.webp`: 390x844 px, 18,850 bytes.
- `06-arsitektur-hybrid-rag.webp`: 1600x1000 px, 171,876 bytes.

All images accompanied by `-preview.webp` lightweight thumbnails for optimal portfolio performance.
