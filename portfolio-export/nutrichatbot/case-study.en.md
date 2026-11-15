# NutriChatBot: Production-Grade Indonesian Nutrition & Diet Assistant with Hybrid RAG

**Author:** Naufal Firdaus  
**Repository:** [https://github.com/victoriavals/NutriChatBot](https://github.com/victoriavals/NutriChatBot)  
**Primary Stack:** Python 3.12, Streamlit, ChromaDB, SQLite, OpenAI SDK / SumoPod AI, GLM-5.3-flash, Docker  
**Date:** November 15, 2026 (Updated with User-Centered UI/UX Architecture)  

---

## 1. Overview

NutriChatBot is an intelligent, domain-specific nutrition and dietary planning assistant engineered to provide verified, culturally grounded nutritional intelligence for Indonesian dietary habits. Rather than relying on generic, ungrounded conversational language models, NutriChatBot integrates a **Dual-Engine Hybrid Retrieval-Augmented Generation (Hybrid RAG)** architecture backed by verified Indonesian food composition data (1,346 foods standard per 100g edible portion, adapted from official Ministry of Health / DKBM records).

Crucially, the system combines rigorous algorithmic retrieval with a **User-Centered Design (UCD)** interface specifically engineered for **everyday laypersons** (home cooks, students, fitness beginners, and elderly users). By eliminating technical jargon, overcoming "blank-canvas" prompt paralysis through interactive suggestion chips, and automating metabolic math into plain-language visual takeaways, NutriChatBot bridges advanced AI engineering with accessible consumer healthcare usability.

Core capabilities include:
- **Tanya Gizi:** Conversational nutritional Q&A with token streaming, interactive prompt chips, and context citation expanders.
- **Perencana Menu Mingguan:** Personalized 7-day meal planning driven by the clinically validated Mifflin-St Jeor metabolic expenditure equation, complete with intuitive BMI categorization and "Isi Piringku" macro visualization.
- **Bandingkan Makanan:** Side-by-side macronutrient analytics powered by Altair visualization and automated "Quick Takeaway" nutritional winners.
- **Masak Apa Hari Ini?:** Pantry-ingredient recipe synthesizer with one-click ingredient combo chips and clinical disease constraint filters (e.g., low-sodium for hypertension, low-sugar for diabetes).
- **Alternatif Sehat:** Healthier ingredient substitution finder with one-click popular ingredient chips.

---

## 2. Problem Statement

Public access to accurate, actionable nutritional information in Indonesia faces four major structural challenges:

1. **Hallucination in General-Purpose LLMs:** Off-the-shelf commercial LLMs frequently hallucinate nutrient values when prompted in Indonesian or queried about localized cuisine. They struggle to differentiate cooking methods (e.g., *tempe goreng* absorbs significant cooking oil, elevating calories to 336 kcal/100g compared to 115 kcal/100g for *tahu goreng*).
2. **Vocabulary Mismatch in Food Search:** Pure semantic vector search often fails to distinguish close taxonomic entities (such as *Beras giling merah* vs. *Beras ketan hitam*), while pure keyword search fails on descriptive queries (such as "makanan berprotein tinggi yang murah dan mudah didapat").
3. **Absence of Clinical Safety Guardrails:** Most AI health chatbots provide uncalibrated medical advice without detecting clinical comorbidities (e.g., diabetes mellitus, chronic kidney disease, hypertension) or providing mandatory medical disclaimers.
4. **"Blank-Canvas" Prompt Paralysis & Cognitive Overload:** Laypersons unfamiliar with prompt engineering struggle when faced with blank textboxes and complex metabolic metrics (e.g., estimating caloric targets without understanding BMR vs. TDEE). Furthermore, consumer AI tools often clutter screens with developer-oriented telemetry (model names, vector counts, endpoints), alienating non-technical users.

---

## 3. Personal Contribution

As the primary author and engineer, Naufal Firdaus conceptualized, implemented, and verified the entire technical lifecycle of NutriChatBot:

- **Hybrid Retrieval Architecture:** Designed and implemented the Weighted Reciprocal Rank Fusion ($wRRF$) algorithm in `services/embeddings.py`, harmonizing dense vector similarity from ChromaDB (`text-embedding-3-small`) with lexical substring/token matching from SQLite.
- **Metabolic Calculation Engine:** Developed `services/nutrition_calc.py`, implementing the Mifflin-St Jeor Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE) formula, complete with rigorous physical bounds validation and macronutrient calorie-ratio distribution.
- **User-Centered Interface & Ergonomics:** Engineered an accessible design system in `app.py`:
  - Injected custom CSS for streamlined container spacing, responsive metric cards, and tactile pill buttons.
  - Developed one-click suggestion chips across all tabs to eliminate blank-canvas cognitive friction.
  - Relegated developer diagnostic telemetry into a collapsible expander drawer, prioritizing a clean, welcoming assistant status for everyday consumers.
  - Implemented automated comparative takeaways in food analysis to deliver instant nutritional insights without requiring users to do mental math.
- **Safety & Guardrails Engineering:** Engineered the clinical comorbidity detection pipeline in `services/llm.py` to intercept high-risk terms (diabetes, ginjal, hipertensi, asam urat) and automatically inject clinical warnings. Implemented the `SensitiveDataFilter` logging filter in `config.py` to prevent credential exposure.
- **Testing & Production Readiness:** Established a 51-test automated suite across unit, integration, and UI testing tiers, reaching 100% pass rates, accompanied by multi-stage Docker containerization and GitHub Actions CI.

---

## 4. Engineering Focus

The engineering effort centered on five critical dimensions:

### A. Dual-Engine Hybrid Retrieval with wRRF
To overcome the limitations of single-paradigm retrieval, NutriChatBot employs a parallel dual-engine search pipeline:
- **Lexical Branch:** Executes indexed keyword scoring in SQLite (`search_nutrition_keyword_ranked`), filtering conversational Indonesian stopwords (`yang`, `dan`, `berapa`, `apakah`) and calculating match scores based on entity name and category exactness.
- **Dense Semantic Branch:** Generates 1536-dimensional embeddings using `text-embedding-3-small` stored in ChromaDB, retrieving documents based on Cosine Distance.
- **Fusion Layer:** Combines rank orders using the Weighted Reciprocal Rank Fusion formula:
  $$wRRF\_Score(d) = \frac{w_{dense}}{k + rank_{dense}(d)} + \frac{w_{lexical}}{k + rank_{lexical}(d)}$$
  with $k=60$, $w_{dense}=0.6$, and $w_{lexical}=0.4$. This ensures exact food terms rank at the top while semantically related items provide surrounding context.

### B. User-Centered Design (UCD) for Non-Technical Users
Engineered specific HCI mechanisms to cater to laypersons:
- **State-Hydrated Prompt Chips:** Integrated one-click suggestion pills that hydrate Streamlit's `st.session_state` before widget instantiation, enabling users to explore common questions (*Kalori Dada Ayam*, *Gizi Nasi Goreng*, *Lemak Sehat Alpukat*) without typing.
- **Visual Nutritional Takeaways:** Programmatically computes macro differentials ($|Cal_1 - Cal_2|$, $|Prot_1 - Prot_2|$) and renders colored callout cards highlighting the "Lower Calorie" and "Higher Protein" options.
- **Demystified Calorie Education:** Translates clinical Mifflin-St Jeor outputs into actionable "Isi Piringku" macro proportions (25% Protein, 25% Healthy Fats, 50% Complex Carbohydrates) paired with color-coded BMI badges (`Ideal 🟢` vs. `Perlu Perhatian 🟡`).

### C. Low-Latency Streaming Inference
Integrated SumoPod AI's OpenAI-compatible endpoint hosting `glm-5.3-flash`. Responses are yielded as real-time chunks through Python generators, drastically reducing Time-to-First-Token (TTFT) and providing a responsive conversational experience inside Streamlit's web worker model.

### D. Zero-Leakage Credential Protection
Implemented custom `logging.Filter` subclasses (`SensitiveDataFilter`) that scan log messages and variable arguments against compiled regular expressions `(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,})`. Any matched API token is instantly replaced with `[REDACTED_API_KEY]`, ensuring security in stdout and container log aggregation.

### E. Comprehensive Verification Pyramid
Built 51 automated test cases spanning:
- Biometric boundary checks and numerical integrity.
- SQL injection resistance with parameterized queries.
- Database idempotency and anomaly detection.
- Embedding sanitization (handling `NaN`, `Inf`, and numpy arrays).
- UI state transitions and error fallbacks via Streamlit `AppTest`.

---

## 5. System Architecture

The NutriChatBot system follows a modular 5-tier architecture:

```
+-----------------------------------------------------------------------------------+
|                            1. PRESENTATION LAYER (UI/UX)                          |
|         Streamlit Multi-Tab Web Application (Desktop 1600x1000 & Mobile 390x844)  |
|  - Custom CSS Design Tokens & Rounded Metric Cards                                |
|  - One-Click Prompt & Recipe Suggestion Chips (Eliminating Blank Canvas)          |
|  - Dedicated Developer Telemetry Drawer (Separated from Layperson UI)             |
|  [Tanya Gizi]   [Masak Apa Hari Ini]   [Perencana Menu]   [Bandingkan Makanan]     |
+------------------------------------------+----------------------------------------+
                                           | User Query & Biometrics
                                           v
+-----------------------------------------------------------------------------------+
|                        2. GUARDRAILS & PREPROCESSING                              |
|  - Indonesian Stopword Stripper & Food Entity Tokenizer                           |
|  - Clinical Risk & Comorbidity Detector (Diabetes, Hipertensi, Ginjal)            |
|  - Input Length & Bounds Validation                                               |
|  - SensitiveDataFilter (API Key Redaction)                                        |
+---------------------+-------------------------------------+-----------------------+
                      | Exact Food Tokens                   | Semantic Intent
                      v                                     v
+-----------------------------------------------------------------------------------+
|                      3. DUAL-ENGINE HYBRID RETRIEVAL                              |
|  +------------------------------------+   +------------------------------------+  |
|  |     Lexical Ranked Search          |   |     Dense Semantic Vector Search   |  |
|  |     SQLite3 `nutrition.db`         |   |     ChromaDB Collection (/chroma)  |  |
|  |     Sub-millisecond keyword match  |   |     text-embedding-3-small (1536d) |  |
|  +-----------------+------------------+   +------------------+-----------------+  |
|                    \                                         /                    |
|                     v                                       v                     |
|           +-------------------------------------------------------+               |
|           |       Weighted Reciprocal Rank Fusion (wRRF)          |               |
|           |       Score(d) = w_d/(k + r_d) + w_l/(k + r_l)        |               |
|           +---------------------------+---------------------------+               |
+---------------------------------------|-------------------------------------------+
                                        | Top-K Formatted Context & System Prompt
                                        v
+-----------------------------------------------------------------------------------+
|                          4. LLM INFERENCE LAYER                                   |
|  - Gateway: SumoPod AI (https://ai.sumopod.com/v1)                                |
|  - Model: GLM-5.3-flash (OpenAI-compatible client)                                |
|  - Output: Real-time token streaming (`st.write_stream`)                          |
|  - Safety: Mandatory Medical Disclaimer appended to every dietary recommendation   |
+---------------------------------------+-------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------------+
|                        5. PERSISTENT STORAGE & DATA                               |
|  - SQLite (nutrition.db): 1,346 structured Indonesian food nutrient records       |
|  - ChromaDB (/chroma): DuckDB/Parquet vector indices with descriptive metadata    |
|  - Raw Source (data/nutrition.csv): Standardized 100g edible portion benchmarks   |
+-----------------------------------------------------------------------------------+
```

---

## 6. Implementation Deep Dive

### Modular Codebase Organization
- `config.py`: Centralized configuration, dataclasses (`AppSettings`), environment loading from `.env`, constants, and the `SensitiveDataFilter`.
- `services/database.py`: Handles CSV validation, anomaly detection for macro-energy plausibility ($|4P + 4C + 9F - Calories| > 50$), SQLite schema management, and ranked keyword search.
- `services/embeddings.py`: Factory functions for OpenAI-compatible embeddings, ChromaDB client management, batch ingestion with `NaN` sanitization, and the `hybrid_retrieve_nutrition` fusion engine.
- `services/nutrition_calc.py`: Pure mathematical service computing Mifflin-St Jeor equations, activity factor multipliers, BMI categorization, and macro splits.
- `services/llm.py`: Prompt synthesis, clinical comorbidity interception, and token streaming generators for Q&A, recipes, weekly plans, and alternatives.
- `app.py`: Streamlit entrypoint coordinating UI styling (`_inject_custom_styles`), interactive chips state-hydration, Altair charting, and layperson-friendly result interpretation.

### Core Mathematical Formulae

#### 1. Mifflin-St Jeor BMR:
$$\text{BMR}_{\text{male}} = 10 \times \text{weight (kg)} + 6.25 \times \text{height (cm)} - 5 \times \text{age (yr)} + 5$$
$$\text{BMR}_{\text{female}} = 10 \times \text{weight (kg)} + 6.25 \times \text{height (cm)} - 5 \times \text{age (yr)} - 161$$

#### 2. Total Daily Energy Expenditure (TDEE):
$$\text{TDEE} = \text{BMR} \times \text{Activity Multiplier}$$
*(Activity Multipliers: Sedentary = 1.2, Light = 1.375, Moderate = 1.55, Very Active = 1.725)*

#### 3. Macro Split Calculation (Deficit/Surplus):
- Target Calories = $\text{TDEE} \pm \Delta_{\text{goal}}$
- Protein: $25\%$ of calories ($4\text{ kcal/g}$)
- Fat: $25\%$ of calories ($9\text{ kcal/g}$)
- Carbohydrates: $50\%$ of calories ($4\text{ kcal/g}$)

---

## 7. Technical Decisions & Trade-Offs

| Decision | Alternative Considered | Rationale |
| :--- | :--- | :--- |
| **User-Centered Design (UCD) with Quick Chips** | Static Empty Inputs | Everyday laypersons face decision paralysis when presented with an empty text box. One-click suggestion pills allow immediate exploration without manual typing, while custom CSS elevates the app to consumer grade. |
| **Separated Developer Telemetry** | Prominent Technical Sidebar Metrics | Displaying LLM names, vector counts, and API endpoints overwhelms non-technical users. Tucking diagnostic info into a collapsible expander preserves developer observability without adding user cognitive load. |
| **Hybrid wRRF Retrieval** | Pure Vector Search (ChromaDB only) | Pure vector search frequently failed on specific Indonesian ingredient names (e.g., distinguishing *Beras giling merah* from *Beras putih*). Combining lexical search ensures 100% precision on exact keywords while retaining semantic context. |
| **SumoPod AI + GLM-5.3-flash** | Direct Google Gemini or OpenAI GPT-4o | SumoPod AI provides high-availability OpenAI-compatible endpoints with GLM-5.3-flash, offering superior reasoning throughput at lower operational costs, verified via local streaming tests. |
| **Mifflin-St Jeor Standard** | Harris-Benedict (1919) | The American Dietetic Association (ADA) confirmed Mifflin-St Jeor is significantly more accurate (within 10% of measured RMR in over 82% of individuals) compared to Harris-Benedict, which overestimates BMR by up to 15%. |
| **Local SQLite + ChromaDB** | Managed Cloud Database (Pinecone / Supabase) | Eliminates external latency and cloud hosting costs for the 1,346-item dataset, enabling completely offline, reproducible local development and containerized portability. |
| **Streamlit Testing (`AppTest`)** | Headless Selenium / Playwright | Streamlit's native `AppTest` enables sub-second headless execution of complete UI state machines inside standard `pytest` without spinning up heavyweight browser processes. |

---

## 8. Evidence and Verification Results

The implementation has been verified through multiple rigorous testing layers:

1. **Automated Pytest Suite:**
   - **51 automated test cases** spanning UI flows, configuration, database operations, embeddings, LLM streaming, and nutrition math.
   - **Status:** 100% PASS (exit code 0, executed in 5.85s).
2. **Dedicated End-to-End QA Suite (`scratch/qa_suite_runner.py`):**
   - **18 test scenarios** covering exact Indonesian food keyword matching, SQL injection resilience with `OR 1=1` and comment syntax, log redaction of API keys, clinical risk warning injection, and BMR mathematical accuracy.
   - **Status:** 18/18 PASS (100% success rate).
3. **Code Quality & Linting:**
   - Evaluated with **Ruff** across all workspace files: `All checks passed!` (zero lint warnings, strict PEP 8 and typing compliance).
4. **Visual UI Verification:**
   - Captured 6 high-fidelity visual assets directly from the running Streamlit server via Chrome DevTools Protocol (CDP), validating updated responsive desktop (1600x1000) and mobile (390x844) layouts featuring the new suggestion chips and design tokens.

---

## 9. Limitations & Ethical Boundary

- **Local Verification Scope:** All tests and benchmarks have been executed in verified local environments and containerized workflows. Real-world public multi-tenant concurrency and production telemetry have not been assessed.
- **Medical Disclaimer Boundary:** NutriChatBot is explicitly an educational dietary assistant, not a licensed medical diagnostic tool. All dietary suggestions and recipes include prominent medical disclaimers urging users with chronic medical conditions to consult a registered clinical dietitian (Sp.GK / RD).
- **Dataset Coverage:** The dataset is bounded to 1,346 foods from standard Indonesian food tables (DKBM). Packaged ultra-processed commercial foods with proprietary barcodes are outside the current database scope.
