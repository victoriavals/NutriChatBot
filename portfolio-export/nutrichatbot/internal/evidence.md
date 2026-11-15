# Technical Evidence & Verification Matrix: NutriChatBot

**Target Project:** NutriChatBot  
**Commit Inspected:** `22eaaff` (and working tree updates)  
**Author:** Naufal Firdaus (`victoriavals` / `naufalfirdaus282@gmail.com`)  
**Audit Date:** 2026-11-15 (Updated with User-Centered UI/UX Engineering)  

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
| **Custom CSS Styling** | `app.py`: Lines 66–115 (`_inject_custom_styles`) | `tests/test_app_ui.py::test_app_all_menus_happy_path_and_indexing` (PASS) |
| **Layperson Prompt Chips** | `app.py`: Lines 205–225 (`chip_ayam`, `chip_nasgor`, `chip_alpukat`, `chip_camilan`) | Streamlit interactive state hydration verified via AppTest & CDP |
| **Comparative Takeaway Cards** | `app.py`: Lines 555–580 (Automated lower calorie & higher protein diff logic) | Visual verification in `03-perbandingan-gizi-visual.webp` |
| **Separated Developer Drawer** | `app.py`: Lines 795–810 (Collapsible telemetry expander in sidebar) | `tests/test_app_ui.py` sidebar assertion checks pass |
| **Dataset Integrity** | `data/nutrition.csv`: 1,346 rows, verified schema & columns | `tests/test_data_integrity.py::test_repository_nutrition_csv_integrity` (PASS) |
| **Household Servings (URT)** | `services/serving_size.py`: Lines 1–180 (`calculate_nutrition_for_serving`) | `tests/test_serving_size.py` (8 tests PASS) |
| **Packaged Foods & Nutri-Score** | `services/database.py`: Lines 380–490 (`search_packaged_foods`, 60 products) | `tests/test_packaged_nutrition.py` (4 tests PASS) |
| **Authentic Regional Recipes** | `services/recipe_service.py` & `services/database.py` (30 authentic recipes) | `tests/test_recipe_service.py` (5 tests PASS) |
| **Brand Logo & Favicon** | `assets/logo.png` & `app.py`: Lines 964–985, 1040–1048 | `tests/test_app_ui.py` (3 UI tests PASS) & Visual Verification |
| **Database Anomaly Checks** | `services/database.py`: Lines 110–145 (`detect_nutrition_anomalies`) | `tests/test_database.py::test_detect_nutrition_anomalies_flags_implausible_macros` (PASS) |

---

## 2. Test Execution Results

### A. Pytest Automated Test Suite
- **Command:** `.venv/Scripts/pytest -v`
- **Total Tests:** 68
- **Passed:** 68
- **Failed:** 0
- **Duration:** 6.35s
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

## 3. Visual & UI Capture Verification (Updated UI)

All visual assets were recaptured live via **Chrome DevTools Protocol (CDP)** against the running instance of `app.py` incorporating the new user-centered design system:
- `01-tanya-gizi-chat.webp`: 1600x1000 px, 28,228 bytes (featuring quick prompt chips and cleaner layout).
- `02-kalkulator-bmr-tdee-dan-menu.webp`: 1600x1000 px, 107,552 bytes (featuring BMI badge and macro visualization).
- `03-perbandingan-gizi-visual.webp`: 1600x1000 px, 100,778 bytes (featuring quick comparison presets and takeaway callouts).
- `04-generator-resep-sehat.webp`: 1600x1000 px, 99,830 bytes (featuring ingredient chips and dietary guides).
- `05-tampilan-mobile-chat.webp`: 390x844 px, 18,864 bytes (responsive mobile layout with accessible chips).
- `06-arsitektur-hybrid-rag.webp`: 1600x1000 px, 171,876 bytes (end-to-end 5-tier architecture diagram).

All images accompanied by `-preview.webp` lightweight thumbnails for optimal portfolio performance.

---

## 4. Cloud Deployment & Public Access Verification

- **Hosting Platform:** Streamlit Community Cloud
- **Public Domain:** [https://nutrichatbot.streamlit.app/](https://nutrichatbot.streamlit.app/)
- **Integration:** Synchronized with GitHub repository `victoriavals/NutriChatBot` on branch `main`.
- **Secrets Management:** Injected via Streamlit Cloud dashboard (`API_KEY`, `BASE_URL`, `MODEL_LLM`, `MODEL_EMBEDDING`).
- **Sleep Mode Policy:** Standard community cloud auto-sleep after prolonged inactivity, with 1-click container wake-up.
