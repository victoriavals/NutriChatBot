"""LLM prompt builders and generation service functions using SumoPod AI / OpenAI."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from config import (
    ALTERNATIVE_MAX_TOKENS,
    ALTERNATIVE_TEMPERATURE,
    DEFAULT_BASE_URL,
    DEFAULT_DAILY_CALORIES,
    DEFAULT_LLM_MODEL,
    DEFAULT_RAG_RESULTS,
    MAX_DAILY_CALORIES,
    MAX_USER_INPUT_LENGTH,
    MENU_MAX_TOKENS,
    MENU_TEMPERATURE,
    MIN_DAILY_CALORIES,
    RAG_MAX_TOKENS,
    RAG_TEMPERATURE,
    RECIPE_MAX_TOKENS,
    RECIPE_TEMPERATURE,
    get_logger,
)
from services.embeddings import retrieve_nutrition_documents

logger = get_logger(__name__)


class InputValidationError(ValueError):
    """Raised when user input fails validation rules."""


class LLMServiceError(Exception):
    """Raised when LLM generation fails."""


MEDICAL_DISCLAIMER: str = (
    "\n\n---\n*Catatan Medis: NutriChatBot adalah asisten edukasi nutrisi dan gaya hidup sehat, "
    "bukan pengganti saran medis profesional, diagnosis, atau terapi dari Dokter Spesialis "
    "Gizi Klinis (Sp.GK) atau Dietisien terdaftar.*"
)

CLINICAL_RISK_KEYWORDS: frozenset[str] = frozenset({
    "ginjal",
    "dialisis",
    "cuci darah",
    "gagal ginjal",
    "diabetes",
    "gula darah",
    "insulin",
    "hipertensi",
    "darah tinggi",
    "asam urat",
    "gout",
    "kolesterol",
    "jantung",
    "alergi",
    "anafilaksis",
    "hamil",
    "menyusui",
})


def detect_clinical_risk(text: str) -> list[str]:
    """Detect presence of critical health condition keywords in user query.

    Args:
        text: Query or input string from user.

    Returns:
        Sorted list of matched clinical keywords.
    """
    lowered = text.lower()
    return [kw for kw in sorted(CLINICAL_RISK_KEYWORDS) if kw in lowered]



def validate_user_input(
    text: str,
    field_name: str = "Input",
    max_length: int = MAX_USER_INPUT_LENGTH,
) -> str:
    """Validate and normalize free-text user input before sending it to an LLM.

    Args:
        text: Raw user input string.
        field_name: Friendly field label for error messages.
        max_length: Maximum allowed character count.

    Returns:
        Trimmed non-empty input string.

    Raises:
        InputValidationError: If text is empty or exceeds max_length.
    """
    if not isinstance(text, str):
        raise InputValidationError(f"{field_name} harus berupa teks.")
    cleaned = text.strip()
    if not cleaned:
        raise InputValidationError(f"{field_name} tidak boleh kosong.")
    if len(cleaned) > max_length:
        raise InputValidationError(
            f"{field_name} terlalu panjang (maksimal {max_length} karakter)."
        )
    return cleaned


def create_llm_client(
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
) -> Any:
    """Configure and return an OpenAI-compatible client for SumoPod AI."""
    if not api_key or not api_key.strip():
        raise ValueError("API_KEY tidak boleh kosong.")

    from openai import OpenAI

    return OpenAI(api_key=api_key.strip(), base_url=base_url)


# Backward-compatible alias
create_gemini_model = create_llm_client


def generate_text(
    model: Any,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    model_name: str = DEFAULT_LLM_MODEL,
) -> str:
    """Execute a single-turn text generation call with shared error handling.

    Supports both OpenAI-compatible client (`chat.completions.create`) and
    mock/legacy objects with `generate_content`.

    Args:
        model: Client instance (e.g. OpenAI client or compatible mock).
        prompt: Prompt text sent to the model.
        temperature: Sampling temperature.
        max_output_tokens: Maximum token budget for the reply.
        model_name: Model identifier (e.g. glm-5.3-flash).

    Returns:
        Generated response text.

    Raises:
        LLMServiceError: If the upstream API call fails or returns empty content.
    """
    try:
        # 1. Standard OpenAI-compatible path (SumoPod AI)
        if hasattr(model, "chat") and hasattr(model.chat, "completions"):
            response = model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            if not response.choices:
                raise LLMServiceError("Model tidak mengembalikan jawaban.")
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            # If content was empty due to token limit consumed by reasoning in reasoning models
            if not text and getattr(choice.message, "reasoning_content", None):
                text = str(choice.message.reasoning_content).strip()

        # 2. Legacy / mock fallback with generate_content
        elif hasattr(model, "generate_content"):
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                },
            )
            text = getattr(response, "text", "") or ""
            text = text.strip()
        else:
            raise TypeError(f"Unsupported client type: {type(model).__name__}")

    except Exception as exc:
        if isinstance(exc, LLMServiceError):
            raise
        logger.exception("LLM generation call failed.")
        raise LLMServiceError(
            "Layanan AI sedang mengalami gangguan. Silakan coba beberapa saat lagi."
        ) from exc

    if not text:
        logger.warning("LLM returned empty response text.")
        raise LLMServiceError("Model tidak mengembalikan jawaban. Silakan coba ulangi pertanyaan.")

    return text


def generate_text_stream(
    model: Any,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    model_name: str = DEFAULT_LLM_MODEL,
) -> Iterator[str]:
    """Execute text generation and yield token chunks as they arrive from upstream.

    Args:
        model: Client instance (e.g. OpenAI client or compatible mock).
        prompt: Prompt text sent to the model.
        temperature: Sampling temperature.
        max_output_tokens: Maximum token budget for the reply.
        model_name: Model identifier (e.g. glm-5.3-flash).

    Yields:
        String chunks of response text.

    Raises:
        LLMServiceError: If the upstream API call fails.
    """
    try:
        if hasattr(model, "chat") and hasattr(model.chat, "completions"):
            stream_resp = model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
                stream=True,
            )
            yielded_any = False
            for chunk in stream_resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None) or ""
                if piece:
                    yielded_any = True
                    yield piece
            if not yielded_any:
                # If stream returned no content (or reasoning consumed it), fallback to non-stream call
                yield generate_text(model, prompt, temperature, max_output_tokens, model_name)
        elif hasattr(model, "generate_content"):
            yield generate_text(model, prompt, temperature, max_output_tokens, model_name)
        else:
            raise TypeError(f"Unsupported client type: {type(model).__name__}")
    except Exception as exc:
        if isinstance(exc, LLMServiceError):
            raise
        logger.exception("LLM streaming call failed.")
        raise LLMServiceError(
            "Layanan AI sedang mengalami gangguan saat streaming. Silakan coba beberapa saat lagi."
        ) from exc



def build_rag_prompt(query: str, contexts: list[str]) -> str:
    """Build grounded RAG prompt combining retrieved nutrition facts with user question."""
    risks = detect_clinical_risk(query)
    risk_clause = ""
    if risks:
        risk_clause = (
            f"\n\n**PERINGATAN KLINIS KHUSUS:** Pengguna menyebutkan kondisi kesehatan berisiko: "
            f"{', '.join(risks)}. Berikan catatan medis hati-hati mengenai batasan nutrisi terkait "
            "(misal batasan kalium/protein untuk ginjal, indeks glikemik untuk diabetes, "
            "natrium untuk darah tinggi, atau purin untuk asam urat), dan tegaskan pentingnya konsultasi dokter.\n"
        )

    if contexts:
        context_block = "\n".join(f"- {doc}" for doc in contexts)
        return (
            "Anda adalah asisten gizi dan resep sehat. Gunakan informasi berikut sebagai "
            "konteks utama untuk menjawab pertanyaan pengguna dalam bahasa Indonesia.\n"
            "Catatan: Data nutrisi pada konteks adalah per 100 gram porsi dapat dimakan.\n"
            "Jika informasi yang diberikan tidak relevan atau tidak cukup untuk menjawab "
            "pertanyaan, nyatakan dengan jujur bahwa Anda tidak memiliki informasi yang cukup.\n"
            f"{risk_clause}\n"
            f"**Konteks Nutrisi:**\n{context_block}\n\n"
            f"**Pertanyaan Pengguna:**\n{query}\n\n"
            "**Jawaban:**"
        )

    return (
        "Anda adalah asisten gizi dan resep sehat. Saya tidak dapat menemukan informasi "
        "nutrisi yang relevan di basis data untuk pertanyaan berikut:\n"
        f"{query}\n"
        f"{risk_clause}\n"
        "Beritahukan pengguna dengan ramah dalam bahasa Indonesia dan tanyakan apakah ada "
        "makanan lain yang ingin dicek."
    )


def build_recipe_prompt(ingredients: str, diet_preference: str = "Biasa") -> str:
    """Build prompt for generating a complete Indonesian healthy recipe from ingredients."""
    pref_str = (
        f" dengan preferensi diet khusus '{diet_preference}'"
        if diet_preference and diet_preference.strip().lower() != "biasa"
        else ""
    )
    return (
        f"Buatkan ide resep makanan sehat dan lengkap{pref_str} berdasarkan bahan-bahan berikut: "
        f"{ingredients}.\n"
        "Sertakan nama resep, estimasi porsi saji, takaran bahan yang wajar, langkah-langkah "
        "pembuatan yang jelas dan berurutan, serta perkiraan kalori & makronutrisi per porsi.\n"
        "Format jawaban dalam bahasa Indonesia yang rapi dan mudah dipahami."
    )



def build_weekly_menu_prompt(daily_calories: int) -> str:
    """Build prompt for generating a 7-day balanced meal plan."""
    return (
        f"Buatkan rencana menu makanan sehat untuk 7 hari (Senin hingga Minggu) dengan target "
        f"rata-rata {daily_calories} kalori per hari.\n"
        "Sertakan sarapan, makan siang, makan malam, dan dua camilan sehat setiap harinya.\n"
        "Berikan variasi makanan bergizi seimbang (sebaiknya relevan dengan bahan makanan "
        "Indonesia) dan format terstruktur per hari dalam bahasa Indonesia."
    )


def build_alternative_prompt(ingredient: str) -> str:
    """Build prompt for suggesting healthier ingredient substitutions."""
    return (
        f"Berikan beberapa alternatif yang lebih sehat untuk bahan makanan '{ingredient}'.\n"
        "Jelaskan secara singkat mengapa setiap alternatif lebih sehat serta cara "
        "penggunaannya sebagai pengganti.\n"
        "Format jawaban dalam bahasa Indonesia yang ringkas dan jelas."
    )


def answer_nutrition_question(
    model: Any,
    query: str,
    chroma_client: Any,
    embedding_function: Any,
    n_results: int = DEFAULT_RAG_RESULTS,
    model_name: str = DEFAULT_LLM_MODEL,
    db_path: str | Path | None = None,
) -> tuple[str, list[str]]:
    """Answer a nutrition question using ChromaDB/Hybrid RAG retrieval and LLM generation."""
    cleaned_query = validate_user_input(query, field_name="Pertanyaan")
    contexts = retrieve_nutrition_documents(
        query=cleaned_query,
        client=chroma_client,
        embedding_function=embedding_function,
        n_results=n_results,
        db_path=db_path,
    )
    prompt = build_rag_prompt(cleaned_query, contexts)
    answer = generate_text(
        model=model,
        prompt=prompt,
        temperature=RAG_TEMPERATURE,
        max_output_tokens=RAG_MAX_TOKENS,
        model_name=model_name,
    )
    return answer, contexts


def answer_nutrition_question_stream(
    model: Any,
    query: str,
    chroma_client: Any,
    embedding_function: Any,
    n_results: int = DEFAULT_RAG_RESULTS,
    model_name: str = DEFAULT_LLM_MODEL,
    db_path: str | Path | None = None,
) -> tuple[Iterator[str], list[str]]:
    """Return a stream generator for the LLM answer alongside retrieved contexts."""
    cleaned_query = validate_user_input(query, field_name="Pertanyaan")
    contexts = retrieve_nutrition_documents(
        query=cleaned_query,
        client=chroma_client,
        embedding_function=embedding_function,
        n_results=n_results,
        db_path=db_path,
    )
    prompt = build_rag_prompt(cleaned_query, contexts)
    stream = generate_text_stream(
        model=model,
        prompt=prompt,
        temperature=RAG_TEMPERATURE,
        max_output_tokens=RAG_MAX_TOKENS,
        model_name=model_name,
    )
    return stream, contexts


def generate_recipe(
    model: Any,
    ingredients: str,
    model_name: str = DEFAULT_LLM_MODEL,
    diet_preference: str = "Biasa",
) -> str:
    """Generate a healthy recipe from comma-separated ingredients."""
    cleaned = validate_user_input(ingredients, field_name="Daftar bahan")
    prompt = build_recipe_prompt(cleaned, diet_preference=diet_preference)
    return generate_text(
        model=model,
        prompt=prompt,
        temperature=RECIPE_TEMPERATURE,
        max_output_tokens=RECIPE_MAX_TOKENS,
        model_name=model_name,
    )


def generate_recipe_stream(
    model: Any,
    ingredients: str,
    model_name: str = DEFAULT_LLM_MODEL,
    diet_preference: str = "Biasa",
) -> Iterator[str]:
    """Generate a healthy recipe as a streaming token iterator."""
    cleaned = validate_user_input(ingredients, field_name="Daftar bahan")
    prompt = build_recipe_prompt(cleaned, diet_preference=diet_preference)
    return generate_text_stream(
        model=model,
        prompt=prompt,
        temperature=RECIPE_TEMPERATURE,
        max_output_tokens=RECIPE_MAX_TOKENS,
        model_name=model_name,
    )


def generate_weekly_menu(
    model: Any,
    daily_calories: int = DEFAULT_DAILY_CALORIES,
    model_name: str = DEFAULT_LLM_MODEL,
) -> str:
    """Generate a 7-day meal plan adhering to calorie bounds."""
    if not isinstance(daily_calories, int) or isinstance(daily_calories, bool):
        raise InputValidationError("Target kalori harus berupa bilangan bulat.")
    if daily_calories < MIN_DAILY_CALORIES or daily_calories > MAX_DAILY_CALORIES:
        raise InputValidationError(
            f"Target kalori harus antara {MIN_DAILY_CALORIES} dan {MAX_DAILY_CALORIES} kcal."
        )
    prompt = build_weekly_menu_prompt(daily_calories)
    return generate_text(
        model=model,
        prompt=prompt,
        temperature=MENU_TEMPERATURE,
        max_output_tokens=MENU_MAX_TOKENS,
        model_name=model_name,
    )


def generate_weekly_menu_stream(
    model: Any,
    daily_calories: int = DEFAULT_DAILY_CALORIES,
    model_name: str = DEFAULT_LLM_MODEL,
) -> Iterator[str]:
    """Generate a 7-day meal plan as a streaming token iterator."""
    if not isinstance(daily_calories, int) or isinstance(daily_calories, bool):
        raise InputValidationError("Target kalori harus berupa bilangan bulat.")
    if daily_calories < MIN_DAILY_CALORIES or daily_calories > MAX_DAILY_CALORIES:
        raise InputValidationError(
            f"Target kalori harus antara {MIN_DAILY_CALORIES} dan {MAX_DAILY_CALORIES} kcal."
        )
    prompt = build_weekly_menu_prompt(daily_calories)
    return generate_text_stream(
        model=model,
        prompt=prompt,
        temperature=MENU_TEMPERATURE,
        max_output_tokens=MENU_MAX_TOKENS,
        model_name=model_name,
    )


def suggest_healthy_alternatives(
    model: Any,
    ingredient: str,
    model_name: str = DEFAULT_LLM_MODEL,
) -> str:
    """Suggest healthier substitutions for a given ingredient."""
    cleaned = validate_user_input(ingredient, field_name="Nama bahan")
    prompt = build_alternative_prompt(cleaned)
    return generate_text(
        model=model,
        prompt=prompt,
        temperature=ALTERNATIVE_TEMPERATURE,
        max_output_tokens=ALTERNATIVE_MAX_TOKENS,
        model_name=model_name,
    )


def suggest_healthy_alternatives_stream(
    model: Any,
    ingredient: str,
    model_name: str = DEFAULT_LLM_MODEL,
) -> Iterator[str]:
    """Suggest healthier substitutions as a streaming token iterator."""
    cleaned = validate_user_input(ingredient, field_name="Nama bahan")
    prompt = build_alternative_prompt(cleaned)
    return generate_text_stream(
        model=model,
        prompt=prompt,
        temperature=ALTERNATIVE_TEMPERATURE,
        max_output_tokens=ALTERNATIVE_MAX_TOKENS,
        model_name=model_name,
    )

