# Missing Information & Unverified Claims: NutriChatBot

**Project:** NutriChatBot  
**Audit Date:** 2026-11-15  
**Auditor:** Technical Writer & Documentation Engineer  

---

## 1. Unverified Production Metrics

The following items **cannot be verified** from codebase or local repository inspection and must NOT be claimed as verified production statistics in portfolio summaries:

| Dimension | Status in Repository | Recommendation for Portfolio |
| :--- | :--- | :--- |
| **Active User Count (MAU / DAU)** | **Belum Terverifikasi** (No analytics SDK or user database table). | State project is an engineered functional system / prototype, not a scaled public consumer app. |
| **Real-World Latency SLA Under Load** | **Belum Terverifikasi** (No load test reports e.g. Locust / k6). | Note sub-second execution on local test suites only; avoid claiming global production p99 SLA. |
| **Cloud Hosting URL** | **Terverifikasi** (Streamlit Community Cloud: https://nutrichatbot.streamlit.app/). | Tautkan URL demo resmi pada portofolio; catat bahwa instance gratis dapat masuk ke mode sleep jika tidak ada kunjungan berkala. |
| **User Retention / Diet Adherence** | **Belum Terverifikasi** (No longitudinal clinical trial data). | Highlight clinical formulation (Mifflin-St Jeor standard), not empirical patient weight loss. |

---

## 2. Boundaries of Dataset & Model Verification

1. **Source Dataset Scope:**
   - The verified dataset consists of 1,346 records from standard Indonesian food composition data.
   - It does not contain barcodes or private commercial brand recipe items.
2. **Third-Party Model Dependency:**
   - Model inference utilizes SumoPod AI (`https://ai.sumopod.com/v1`) with model `glm-5.3-flash`.
   - Embeddings utilize `text-embedding-3-small`.
   - External API availability is dependent on network connectivity and valid SumoPod AI credits.
3. **Clinical Responsibility:**
   - All AI-generated dietary recommendations and meal plans explicitly append `MEDICAL_DISCLAIMER`.
   - The application is an educational decision-support tool, not a regulated Medical Device Software (SaMD).
