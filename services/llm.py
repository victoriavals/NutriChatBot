"""Gemini LLM prompt builders and generation service functions."""

from __future__ import annotations

from typing import Any

from config import (
    ALTERNATIVE_MAX_TOKENS,
    ALTERNATIVE_TEMPERATURE,
    DEFAULT_DAILY_CALORIES,
    DEFAULT_GEMINI_MODEL,
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
    """Raised when Gemini generation fails."""


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


def create_gemini_model(
    api_key: str,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Any:
    """Configure google-generativeai SDK and return a GenerativeModel instance."""
    if not api_key or not api_key.strip():
        raise ValueError("GEMINI_API_KEY tidak boleh kosong.")

    import google.generativeai as genai

    genai.configure(api_key=api_key.strip())
    return genai.GenerativeModel(model_name)


def generate_text(
    model: Any,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> str:
    """Execute a single-turn Gemini text generation call with shared error handling.

    Args:
        model: Configured Gemini GenerativeModel (or compatible mock).
        prompt: Prompt text sent to the model.
        temperature: Sampling temperature.
        max_output_tokens: Maximum token budget for the reply.

    Returns:
        Generated response text.

    Raises:
        LLMServiceError: If the upstream API call fails or returns empty content.
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )
    except Exception as exc:
        logger.exception("Gemini generation call failed.")
        raise LLMServiceError(
            "Layanan AI sedang mengalami gangguan. Silakan coba beberapa saat lagi."
        ) from exc

    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        logger.warning("Gemini returned empty response text.")
        raise LLMServiceError("Model tidak mengembalikan jawaban. Silakan coba ulangi pertanyaan.")

    return text.strip()


def build_rag_prompt(query: str, contexts: list[str]) -> str:
    """Build grounded RAG prompt combining retrieved nutrition facts with user question."""
    if contexts:
        context_block = "\n".join(f"- {doc}" for doc in contexts)
        return (
            "Anda adalah asisten gizi dan resep sehat. Gunakan informasi berikut sebagai "
            "konteks utama untuk menjawab pertanyaan pengguna dalam bahasa Indonesia.\n"
            "Jika informasi yang diberikan tidak relevan atau tidak cukup untuk menjawab "
            "pertanyaan, nyatakan dengan jujur bahwa Anda tidak memiliki informasi yang cukup.\n\n"
            f"**Konteks Nutrisi:**\n{context_block}\n\n"
            f"**Pertanyaan Pengguna:**\n{query}\n\n"
            "**Jawaban:**"
        )

    return (
        "Anda adalah asisten gizi dan resep sehat. Saya tidak dapat menemukan informasi "
        "nutrisi yang relevan di basis data untuk pertanyaan berikut:\n"
        f"{query}\n"
        "Beritahukan pengguna dengan ramah dalam bahasa Indonesia dan tanyakan apakah ada "
        "makanan lain yang ingin dicek."
    )


def build_recipe_prompt(ingredients: str) -> str:
    """Build prompt for generating a complete Indonesian healthy recipe from ingredients."""
    return (
        f"Buatkan ide resep makanan sehat dan lengkap berdasarkan bahan-bahan berikut: "
        f"{ingredients}.\n"
        "Sertakan nama resep, daftar bahan lengkap dengan takaran wajar, dan langkah-langkah "
        "pembuatan yang jelas dan berurutan.\n"
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
) -> tuple[str, list[str]]:
    """Answer a nutrition question using ChromaDB RAG retrieval and Gemini generation."""
    cleaned_query = validate_user_input(query, field_name="Pertanyaan")
    contexts = retrieve_nutrition_documents(
        query=cleaned_query,
        client=chroma_client,
        embedding_function=embedding_function,
        n_results=n_results,
    )
    prompt = build_rag_prompt(cleaned_query, contexts)
    answer = generate_text(
        model=model,
        prompt=prompt,
        temperature=RAG_TEMPERATURE,
        max_output_tokens=RAG_MAX_TOKENS,
    )
    return answer, contexts


def generate_recipe(model: Any, ingredients: str) -> str:
    """Generate a healthy recipe from comma-separated ingredients."""
    cleaned = validate_user_input(ingredients, field_name="Daftar bahan")
    prompt = build_recipe_prompt(cleaned)
    return generate_text(
        model=model,
        prompt=prompt,
        temperature=RECIPE_TEMPERATURE,
        max_output_tokens=RECIPE_MAX_TOKENS,
    )


def generate_weekly_menu(
    model: Any,
    daily_calories: int = DEFAULT_DAILY_CALORIES,
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
    )


def suggest_healthy_alternatives(model: Any, ingredient: str) -> str:
    """Suggest healthier substitutions for a given ingredient."""
    cleaned = validate_user_input(ingredient, field_name="Nama bahan")
    prompt = build_alternative_prompt(cleaned)
    return generate_text(
        model=model,
        prompt=prompt,
        temperature=ALTERNATIVE_TEMPERATURE,
        max_output_tokens=ALTERNATIVE_MAX_TOKENS,
    )

