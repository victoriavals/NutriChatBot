"""Unit tests for LLM prompt builders and generation services."""

from __future__ import annotations

import pytest

from config import MAX_USER_INPUT_LENGTH
from services.llm import (
    InputValidationError,
    LLMServiceError,
    answer_nutrition_question,
    answer_nutrition_question_stream,
    build_rag_prompt,
    create_llm_client,
    detect_clinical_risk,
    generate_recipe,
    generate_recipe_stream,
    generate_text_stream,
    generate_weekly_menu,
    suggest_healthy_alternatives,
)
from tests.conftest import FakeChromaClient, FakeGeminiModel, FakeOpenAIClient


def test_create_llm_client_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="API_KEY"):
        create_llm_client("   ")


def test_generate_recipe_validates_and_calls_model() -> None:
    client = FakeOpenAIClient(reply="Resep Tumis Brokoli Ayam")
    result = generate_recipe(client, "ayam, brokoli", model_name="glm-5.3-flash")

    assert result == "Resep Tumis Brokoli Ayam"
    assert len(client.calls) == 1
    assert "ayam, brokoli" in client.calls[0]["messages"][0]["content"]
    assert client.calls[0]["model"] == "glm-5.3-flash"
    assert client.calls[0]["max_tokens"] == 2048


def test_generate_recipe_rejects_blank_or_oversized_input() -> None:
    client = FakeOpenAIClient()
    with pytest.raises(InputValidationError, match="tidak boleh kosong"):
        generate_recipe(client, "   ")

    with pytest.raises(InputValidationError, match="terlalu panjang"):
        generate_recipe(client, "a" * (MAX_USER_INPUT_LENGTH + 1))


def test_generate_weekly_menu_validates_calorie_bounds() -> None:
    client = FakeOpenAIClient(reply="Menu 7 Hari")
    assert generate_weekly_menu(client, 2000) == "Menu 7 Hari"

    with pytest.raises(InputValidationError, match="antara 1000 dan 5000"):
        generate_weekly_menu(client, 500)

    with pytest.raises(InputValidationError, match="bilangan bulat"):
        generate_weekly_menu(client, True)  # type: ignore[arg-type]


def test_suggest_healthy_alternatives_happy_path() -> None:
    client = FakeOpenAIClient(reply="Gunakan madu atau stevia")
    result = suggest_healthy_alternatives(client, "gula pasir")
    assert "madu" in result
    assert "gula pasir" in client.calls[0]["messages"][0]["content"]


def test_llm_errors_do_not_leak_sensitive_details() -> None:
    failing_client = FakeOpenAIClient(should_fail=True)
    with pytest.raises(LLMServiceError) as exc_info:
        generate_recipe(failing_client, "tempe")

    assert "sk-FakeSecret" not in str(exc_info.value)


def test_empty_model_response_raises_llm_service_error() -> None:
    empty_client = FakeOpenAIClient(reply="   ")
    with pytest.raises(LLMServiceError, match="tidak mengembalikan jawaban"):
        suggest_healthy_alternatives(empty_client, "garam")


def test_answer_nutrition_question_integrates_rag_and_llm() -> None:
    chroma_client = FakeChromaClient()
    col = chroma_client.get_or_create_collection("nutrition", object())
    col.add(
        documents=["Nasi Goreng has 250 calories, 8g protein, 9g fat, 34g carbohydrate."],
        metadatas=[{"name": "Nasi Goreng"}],
        ids=["doc_0"],
    )
    llm_client = FakeOpenAIClient(reply="Nasi goreng mengandung sekitar 250 kalori.")

    answer, contexts = answer_nutrition_question(
        model=llm_client,
        query="Berapa kalori nasi goreng?",
        chroma_client=chroma_client,
        embedding_function=object(),
    )

    assert "250 kalori" in answer
    assert len(contexts) == 1
    assert "Nasi Goreng has 250 calories" in llm_client.calls[0]["messages"][0]["content"]


def test_legacy_gemini_model_mock_compatibility() -> None:
    legacy_model = FakeGeminiModel(reply="Legacy mock response")
    result = generate_recipe(legacy_model, "tahu")
    assert result == "Legacy mock response"


def test_build_rag_prompt_without_context_uses_fallback_message() -> None:
    prompt = build_rag_prompt("kalori boba", [])
    assert "tidak dapat menemukan informasi" in prompt


def test_detect_clinical_risk_identifies_keywords() -> None:
    text = "Saya menderita diabetes dan gagal ginjal stadium 3, boleh makan alpukat?"
    risks = detect_clinical_risk(text)
    assert "diabetes" in risks
    assert "gagal ginjal" in risks
    assert "ginjal" in risks

    # Safe query
    assert detect_clinical_risk("Berapa kalori apel?") == []


def test_build_rag_prompt_injects_clinical_warning() -> None:
    prompt = build_rag_prompt("Saya ada diabetes, boleh makan nasi putih?", ["Nasi putih mengandung energi..."])
    assert "PERINGATAN KLINIS KHUSUS" in prompt
    assert "diabetes" in prompt


def test_generate_text_stream_yields_token_pieces() -> None:
    client = FakeOpenAIClient(reply="Resep Ayam Panggang")
    chunks = list(generate_text_stream(client, "prompt", temperature=0.7, max_output_tokens=100))
    full_text = "".join(chunks)
    assert full_text == "Resep Ayam Panggang"
    assert len(chunks) > 1  # Verify chunked streaming


def test_answer_nutrition_question_stream_returns_stream_and_contexts() -> None:
    chroma_client = FakeChromaClient()
    col = chroma_client.get_or_create_collection("nutrition", object())
    col.add(
        documents=["Ayam goreng dada mengandung energi 290.0 kkal..."],
        metadatas=[{"name": "Ayam goreng dada"}],
        ids=["doc_1"],
    )
    llm_client = FakeOpenAIClient(reply="Dada ayam memiliki protein tinggi.")

    stream, contexts = answer_nutrition_question_stream(
        model=llm_client,
        query="Berapa protein dada ayam?",
        chroma_client=chroma_client,
        embedding_function=object(),
    )
    assert len(contexts) == 1
    assert "".join(list(stream)) == "Dada ayam memiliki protein tinggi."


def test_generate_recipe_with_diet_preference() -> None:
    client = FakeOpenAIClient(reply="Resep Vegan Tempe")
    result = generate_recipe(client, "tempe, bayam", diet_preference="Vegetarian")
    assert result == "Resep Vegan Tempe"
    assert "Vegetarian" in client.calls[0]["messages"][0]["content"]

    stream = generate_recipe_stream(client, "tempe, bayam", diet_preference="Vegetarian")
    assert "".join(list(stream)) == "Resep Vegan Tempe"


