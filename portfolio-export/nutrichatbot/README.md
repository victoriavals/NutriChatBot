# NutriChatBot Portfolio Export Package

This directory contains the verified portfolio documentation and visual assets for **NutriChatBot**, authored by **Naufal Firdaus**. It is structured for direct ingestion by automated portfolio website builders and content management systems.

---

## 1. Directory Structure

```
nutrichatbot/
├── README.md                  # This integration guide
├── project.json               # Bilingual structured project metadata & schema
├── case-study.en.md           # Full engineering case study (English)
├── case-study.id.md           # Full engineering case study (Indonesian)
├── assets-manifest.json       # SHA-256, dimensions, captions, and alt-text manifest
├── assets/                    # Optimized WebP visual assets & preview thumbnails
│   ├── 01-tanya-gizi-chat.webp
│   ├── 01-tanya-gizi-chat-preview.webp
│   ├── 02-kalkulator-bmr-tdee-dan-menu.webp
│   ├── 02-kalkulator-bmr-tdee-dan-menu-preview.webp
│   ├── 03-perbandingan-gizi-visual.webp
│   ├── 03-perbandingan-gizi-visual-preview.webp
│   ├── 04-generator-resep-sehat.webp
│   ├── 04-generator-resep-sehat-preview.webp
│   ├── 05-tampilan-mobile-chat.webp
│   ├── 05-tampilan-mobile-chat-preview.webp
│   ├── 06-arsitektur-hybrid-rag.webp
│   └── 06-arsitektur-hybrid-rag-preview.webp
└── internal/                  # Engineering audit & provenance records
    ├── evidence.md            # Traceability matrix mapping claims to lines & tests
    ├── missing-information.md # Unverified metrics boundary (zero fabricated data)
    └── privacy-review.md      # Zero-leakage secret & PII audit confirmation
```

---

## 2. Integration Instructions for Portfolio Agents

1. **Card Showcase:**
   - Parse `project.json` for `title`, `domain`, `stack`, and `summary` (bilingual `en` and `id`).
   - Use `assets/01-tanya-gizi-chat-preview.webp` as the lightweight card thumbnail.
2. **Project Detail Page / Case Study:**
   - Use `case-study.en.md` for English language pages and `case-study.id.md` for Indonesian language pages.
   - Embed visual assets from `assets/` referencing the bilingual alt-text and captions in `assets-manifest.json`.
   - Feature `assets/01-tanya-gizi-chat.webp` as the primary hero image.
   - Embed `assets/06-arsitektur-hybrid-rag.webp` in Section 5 (Architecture).
3. **Internal Documentation:**
   - Keep files in `internal/` for internal verification, provenance tracking, and interview preparation. They do not need to be published on public web pages.

---

## 3. Privacy & Security Certification

This package has undergone a strict privacy audit:
- **Zero API Keys:** No Google AI Studio (`AIza...`) or SumoPod/OpenAI (`sk-...`) tokens exist in any file.
- **Zero PII:** No phone numbers, student IDs, or personal residential addresses.
- **Safe for Public Exhibition:** Ready for public deployment on personal portfolios and GitHub.
