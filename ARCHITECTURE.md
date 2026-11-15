# NutriChatBot: System Architecture & Technical Specifications

**Author:** Naufal Firdaus  
**System Version:** 2.1 (Hybrid RAG + SumoPod GLM-5.3-flash)  
**Date:** November 15, 2026  

---

## 1. Executive Technical Summary

NutriChatBot is an AI-powered conversational nutrition and meal-planning system engineered for Indonesian culinary contexts. It resolves the dual challenges of language model hallucination on local food composition tables and cognitive decision fatigue in non-technical users.

The system is architected around five decoupled tiers:
1. **Presentation & Interaction Layer (UI/UX)**: Streamlit web application with custom design tokens, responsive viewports, state-hydrated quick prompt chips, and separate developer diagnostic drawers.
2. **Safety & Guardrails Engine**: Indonesian stopword strippers, food entity tokenizers, comorbidity clinical risk detectors, and regex-based sensitive log filters.
3. **Dual-Engine Hybrid Retrieval & Fusion Core**: Parallel lexical scoring in SQLite and dense semantic search in ChromaDB (`text-embedding-3-small`), unified by Weighted Reciprocal Rank Fusion ($wRRF$).
4. **Metabolic Calculation Engine**: Validated biometric calculations using the clinical Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and Total Daily Energy Expenditure (TDEE).
5. **LLM Inference & Streaming Layer**: OpenAI-compatible client interfacing with SumoPod AI (`glm-5.3-flash`), streaming token-by-token responses via Python generators with mandatory medical disclaimers.

---

## 2. Architectural Blueprint

```text
                                 [ End User ]
                                      │
                    ┌─────────────────▼─────────────────┐
                    │      1. PRESENTATION LAYER        │
                    │      (Streamlit Web Server)       │
                    │  - Custom CSS & Responsive Grid   │
                    │  - Quick Prompt & Recipe Chips    │
                    │  - Collapsible Developer Drawer   │
                    └─────────────────┬─────────────────┘
                                      │
                         User Query & Biometrics
                                      │
                    ┌─────────────────▼─────────────────┐
                    │    2. GUARDRAILS & PROCESSOR      │
                    │  - Tokenizer & Stopword Stripper  │
                    │  - Clinical Comorbidity Detector  │
                    │  - SensitiveDataFilter Logger     │
                    └─────────┬───────────────┬─────────┘
                              │               │
                    Food Entity Tokens   Semantic Intent
                              │               │
       ┌──────────────────────▼──┐         ┌──▼──────────────────────┐
       │   3A. Lexical Engine    │         │    3B. Dense Engine     │
       │   (SQLite nutrition.db) │         │   (ChromaDB Collection) │
       │   - Parameterized LIKE  │         │   - text-embedding-3-   │
       │   - Multi-term matching │         │     small (1536 dims)   │
       │   - Exact name scoring  │         │   - Cosine Distance     │
       └──────────────┬──────────┘         └──┬──────────────────────┘
                      │                       │
                 Ranked List             Ranked List
                      └───────────┬───────────┘
                                  │
                    ┌─────────────▼─────────────────────┐
                    │      3C. FUSION LAYER (wRRF)      │
                    │  Score(d) = w_d/(k+r_d)+w_l/(k+r_l│
                    │  (k=60, w_d=0.6, w_l=0.4)         │
                    └─────────────┬─────────────────────┘
                                  │
                       Top-K Nutritional Context
                                  │
                    ┌─────────────▼─────────────────────┐
                    │      4. LLM INFERENCE LAYER       │
                    │  SumoPod AI Gateway (OpenAI SDK)  │
                    │  Model: GLM-5.3-flash             │
                    │  Real-Time Token Streaming        │
                    │  + Mandatory Medical Disclaimer   │
                    └─────────────┬─────────────────────┘
                                  │
                         Rendered Output
                                  │
                    ┌─────────────▼─────────────────────┐
                    │      5. METABOLIC ENGINE          │
                    │  Mifflin-St Jeor (BMR & TDEE)     │
                    │  BMI Categorization & Macro Split │
                    └───────────────────────────────────┘
```

---

## 3. Retrieval Methodology & Mathematical Derivations

### A. Dual-Engine Justification
Traditional single-paradigm search systems fail in domain-specific dietary applications:
- **Dense Vector Search Failure Mode**: High semantic overlap between distinct ingredients (e.g., *Beras giling putih* vs *Beras merah*) causes dense vector models to confuse nutritional properties.
- **Lexical Search Failure Mode**: Inability to map natural conversational phrases (e.g., *"lauk pauk tinggi protein untuk defisit kalori"*) to specific food records.

### B. Weighted Reciprocal Rank Fusion ($wRRF$)
NutriChatBot merges rank positions rather than arbitrary raw scores:

$$wRRF(d) = \sum_{m \in M} \frac{w_m}{k + r_m(d)}$$

Where:
- $M = \{\text{dense}, \text{lexical}\}$
- $k = 60$ (smoothing constant preventing top ranks from disproportionately dominating)
- $w_{\text{dense}} = 0.6$
- $w_{\text{lexical}} = 0.4$
- $r_m(d) \in [1, N]$ represents the 1-based rank position of document $d$ in search engine $m$. If document $d$ does not appear in engine $m$, its rank term is evaluated as zero.

### C. Context Assembly
The top $K$ documents (default $K=3$) are formatted into standardized nutritional summaries injected into the system prompt:
```text
[Katalog Gizi] Nama: {name} | Kategori: {category} | Porsi: 100g BDD
Energi: {calories} kkal | Protein: {proteins}g | Lemak: {fat}g | Karbohidrat: {carbohydrate}g
Serat: {fiber}g | Kalsium: {calcium}mg | Besi: {iron}mg | Natrium: {sodium}mg
```

---

## 4. Metabolic & Biometric Formulation

NutriChatBot implements the **Mifflin-St Jeor Equation**, selected based on recommendations by the *American Dietetic Association* (ADA) for highest clinical accuracy across diverse BMI ranges.

### A. Basal Metabolic Rate (BMR)
$$\text{BMR}_{\text{male}} = 10W + 6.25H - 5A + 5$$
$$\text{BMR}_{\text{female}} = 10W + 6.25H - 5A - 161$$

Where:
- $W$ = Body weight in kilograms ($25.0 \le W \le 250.0$)
- $H$ = Stature in centimeters ($50.0 \le H \le 250.0$)
- $A$ = Chronological age in completed years ($10 \le A \le 120$)

### B. Total Daily Energy Expenditure (TDEE)
$$\text{TDEE} = \text{BMR} \times \alpha$$

Activity coefficient ($\alpha$):
- **Sedentary** ($\alpha = 1.200$): Minimal movement, desk work.
- **Lightly Active** ($\alpha = 1.375$): Light exercise 1–3 days/week.
- **Moderately Active** ($\alpha = 1.550$): Moderate exercise 3–5 days/week.
- **Very Active** ($\alpha = 1.725$): Heavy exercise/physical labor 6–7 days/week.

### C. Caloric Target & Macronutrient Split
- **Weight Loss Goal**: $\text{Target} = \max(\text{TDEE} - 500, \text{BMR})$ (preserving safe deficit limits).
- **Maintenance Goal**: $\text{Target} = \text{TDEE}$.
- **Weight Gain Goal**: $\text{Target} = \text{TDEE} + 350$.

Macronutrient calorie allocations:
- **Protein**: 25% of target calories ($4\text{ kcal/g}$) $\rightarrow P_g = \frac{0.25 \times \text{Target}}{4}$
- **Fat**: 25% of target calories ($9\text{ kcal/g}$) $\rightarrow F_g = \frac{0.25 \times \text{Target}}{9}$
- **Carbohydrates**: 50% of target calories ($4\text{ kcal/g}$) $\rightarrow C_g = \frac{0.50 \times \text{Target}}{4}$

---

## 5. Security, Guardrails, & Clinical Boundaries

### A. Clinical Comorbidity Interception
In `services/llm.py`, incoming queries are inspected for high-risk clinical keywords:
```python
CLINICAL_RISK_KEYWORDS = {
    "diabetes": "Gunakan indeks glikemik rendah dan batasi karbohidrat sederhana.",
    "hipertensi": "Batasi natrium/garam dapur di bawah 2000mg/hari.",
    "ginjal": "Waspadai asupan protein berlebih dan batasi fosfor/kalium.",
    "asam urat": "Hindari bahan tinggi purin seperti jeroan dan ekstrak ragi.",
}
```
When triggered, a dedicated clinical advisory banner is injected into the model's system prompt instructing conservative, evidence-based recommendations.

### B. Logging Secret Redaction (`SensitiveDataFilter`)
All logging calls pass through a centralized filter:
```python
_API_KEY_PATTERN = re.compile(r"(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,})")
```
Every matched credential token in log messages or arguments is redacted into `[REDACTED_API_KEY]`.

### C. SQL Injection Defense
All SQLite operations in `services/database.py` utilize parameterized queries:
```python
cursor.execute("SELECT ... WHERE name LIKE ? ESCAPE '\'", (escaped_param,))
```
Wildcard characters (`%` and `_`) are escaped with a backslash before parameter binding.

---

## 6. Verification & Testing Pyramid

The project enforces strict automated verification:

| Testing Tier | Tooling | Coverage |
| :--- | :--- | :--- |
| **Unit Testing (60%)** | `pytest` | Pure math, formula bounds, stopword stripping, regex sanitization |
| **Integration Testing (30%)** | `pytest`, SQLite, ChromaDB | End-to-end RAG retrieval, wRRF ranking, LLM client mocks |
| **UI State Testing (10%)** | Streamlit `AppTest` | Multi-tab navigation, query parameter handling, session state |
| **Linter / Formatter** | `ruff` | PEP 8, flake8, imports formatting, zero lint warnings |

---

## 7. Data Storage Schema

### SQLite Table Schema (`nutrition.db`)
```sql
CREATE TABLE IF NOT EXISTS nutrition (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    calories REAL NOT NULL,
    proteins REAL NOT NULL,
    fat REAL NOT NULL,
    carbohydrate REAL NOT NULL,
    fiber REAL,
    calcium REAL,
    phosphorus REAL,
    iron REAL,
    sodium REAL,
    potassium REAL,
    copper REAL,
    zinc REAL,
    retinol REAL,
    b_carotene REAL,
    thiamin REAL,
    riboflavin REAL,
    niacin REAL,
    vitamin_c REAL,
    image TEXT,
    description TEXT
);
CREATE INDEX IF NOT EXISTS idx_nutrition_name ON nutrition(name);
CREATE INDEX IF NOT EXISTS idx_nutrition_category ON nutrition(category);

### SQLite Table Schema (`packaged_nutrition`)
```sql
CREATE TABLE IF NOT EXISTS packaged_nutrition (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    brand TEXT NOT NULL,
    category TEXT NOT NULL,
    serving_size_g REAL NOT NULL,
    serving_unit TEXT NOT NULL,
    calories REAL NOT NULL,
    proteins REAL NOT NULL,
    fat REAL NOT NULL,
    carbohydrate REAL NOT NULL,
    saturated_fat REAL,
    sugars REAL,
    sodium_mg REAL,
    nutri_score TEXT,
    description TEXT
);
CREATE INDEX IF NOT EXISTS idx_packaged_name ON packaged_nutrition(name);
CREATE INDEX IF NOT EXISTS idx_packaged_brand ON packaged_nutrition(brand);
CREATE INDEX IF NOT EXISTS idx_packaged_category ON packaged_nutrition(category);
```

### SQLite Table Schema (`recipes`)
```sql
CREATE TABLE IF NOT EXISTS recipes (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    origin_region TEXT NOT NULL,
    category TEXT NOT NULL,
    prep_time_minutes INTEGER NOT NULL,
    servings INTEGER NOT NULL,
    ingredients_json TEXT NOT NULL,
    instructions_text TEXT NOT NULL,
    calories_per_serving REAL NOT NULL,
    protein_g REAL NOT NULL,
    fat_g REAL NOT NULL,
    carb_g REAL NOT NULL,
    health_tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(name);
CREATE INDEX IF NOT EXISTS idx_recipes_category ON recipes(category);
CREATE INDEX IF NOT EXISTS idx_recipes_origin ON recipes(origin_region);
```
```
