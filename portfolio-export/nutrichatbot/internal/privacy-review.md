# Privacy, Security, & Publication Review: NutriChatBot

**Target Export:** `portfolio-export/nutrichatbot/`  
**Review Date:** 2026-11-15  
**Compliance Standard:** Strict Zero-Leakage Privacy Review  

---

## 1. Sensitive Data Audit Checklist

| Item | Checked? | Result | Notes |
| :--- | :---: | :---: | :--- |
| **`.env` and Secrets** | [x] | **CLEAN** | Excluded completely from the export bundle. Only `.env.example` is discussed in docs. |
| **API Keys (Google / SumoPod)** | [x] | **CLEAN** | No active tokens starting with `AIza...` or `sk-...` are present in exported files. |
| **Personally Identifiable Info (PII)** | [x] | **CLEAN** | No phone numbers, national IDs (NIK/NIM), or residential addresses are present. |
| **Personal Absolute File Paths** | [x] | **CLEAN** | All internal paths sanitized to repository root relative paths. |
| **Git & Cache Artifacts** | [x] | **CLEAN** | No `.git`, `.pytest_cache`, `.ruff_cache`, or `__pycache__` included in the export. |
| **Image Metadata (EXIF)** | [x] | **CLEAN** | All `.webp` images were freshly generated from rendered UI canvas; EXIF sanitized. |

---

## 2. In-Code Security & Privacy Protections

### A. Automatic Log Redaction (`config.py`)
`NutriChatBot` employs an active `SensitiveDataFilter` registered to root logging handlers:
```python
_API_KEY_PATTERN = re.compile(r"(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,})")
```
Tested and verified: even if an unhandled error passes an API client string with tokens into logger calls, the emitted log line automatically replaces the key with `[REDACTED_API_KEY]`.

### B. Parameterized SQL Execution (`services/database.py`)
All database search calls use parameterized SQLite syntax:
```python
cursor.execute("SELECT ... WHERE name LIKE ? ESCAPE '\'", (escaped_param,))
```
Tested against common SQL injection payloads (`' OR 1=1 --`, `UNION SELECT`) with zero escape.

---

## 3. Publication Recommendation

The package `portfolio-export/nutrichatbot/` and its resulting archive `portfolio-nutrichatbot-2026-11-15.zip` are **fully verified and safe for public portfolio display**. No confidential secrets, sensitive personal data, or proprietary non-open-source assets are present.
