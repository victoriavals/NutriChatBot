"""Unit tests for Gemini LLM prompt builders and generation services."""

from __future__ import annotations

import pytest

from config import MAX_USER_INPUT_LENGTH
from services.llm import (
    InputValidationError,
    LLMServiceError,
    answer_nutrition_question,
    build_rag_prompt,
    generate_recipe,
    generate_weekly_menu,
    suggest_healthy_alternatives,
)
from tests.conftest import FakeChromaClient, FakeGeminiModel


def test_generate_recipe_validates_and_calls_model() -> None:
    model = FakeGeminiModel(reply="Resep Tumis Brokoli Ayam")
    result = generate_recipe(model, "ayam, brokoli")

    assert result == "Resep Tumis Brokoli Ayam"
    assert len(model.calls) == 1
    assert "ayam, brokoli" in model.calls[0]["prompt"]
    assert model.calls[0]["generation_config"]["max_output_tokens"] == 1024


def test_generate_recipe_rejects_blank_or_oversized_input() -> None:
    model = FakeGeminiModel()
    with pytest.raises(InputValidationError, match="tidak boleh kosong"):
        generate_recipe(model, "   ")

    with pytest.raises(InputValidationError, match="terlalu panjang"):
        generate_recipe(model, "a" * (MAX_USER_INPUT_LENGTH + 1))


def test_generate_weekly_menu_validates_calorie_bounds() -> None:
    model = FakeGeminiModel(reply="Menu 7 Hari")
    assert generate_weekly_menu(model, 2000) == "Menu 7 Hari"

    with pytest.raises(InputValidationError, match="antara 1000 dan 5000"):
        generate_weekly_menu(model, 500)

    with pytest.raises(InputValidationError, match="bilangan bulat"):
        generate_weekly_menu(model, True)  # type: ignore[arg-type]


def test_suggest_healthy_alternatives_happy_path() -> None:
    model = FakeGeminiModel(reply="Gunakan madu atau stevia")
    result = suggest_healthy_alternatives(model, "gula pasir")
    assert "madu" in result
    assert "gula pasir" in model.calls[0]["prompt"]


def test_llm_errors_do_not_leak_sensitive_details() -> None:
    failing_model = FakeGeminiModel(should_fail=True)
    with pytest.raises(LLMServiceError) as exc_info:
        generate_recipe(failing_model, "tempe")

    assert "AIza" not in str(exc_info.value)


def test_empty_model_response_raises_llm_service_error() -> None:
    empty_model = FakeGeminiModel(reply="   ")
    with pytest.raises(LLMServiceError, match="tidak mengembalikan jawaban"):
        suggest_healthy_alternatives(empty_model, "garam")


def test_answer_nutrition_question_integrates_rag_and_llm() -> None:
    client = FakeChromaClient()
    col = client.get_or_create_collection("nutrition", object())
    col.add(
        documents=["Nasi Goreng has 250 calories, 8g protein, 9g fat, 34g carbohydrate."],
        metadatas=[{"name": "Nasi Goreng"}],
        ids=["doc_0"],
    )
    model = FakeGeminiModel(reply="Nasi goreng mengandung sekitar 250 kalori.")

    answer, contexts = answer_nutrition_question(
        model=model,
        query="Berapa kalori nasi goreng?",
        chroma_client=client,
        embedding_function=object(),
    )

    assert "250 kalori" in answer
    assert len(contexts) == 1
    assert "Nasi Goreng has 250 calories" in model.calls[0]["prompt"]


def test_build_rag_prompt_without_context_uses_fallback_message() -> None:
    prompt = build_rag_prompt("kalori boba", [])
    assert "tidak dapat menemukan informasi" in prompt

