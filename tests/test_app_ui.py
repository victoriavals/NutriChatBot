"""Streamlit AppTest integration tests for app.py UI flows."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import AppSettings, ConfigurationError
from tests.conftest import FakeChromaClient, FakeGeminiModel

APP_TEST_TIMEOUT = 15


@pytest.fixture(autouse=True)
def clear_streamlit_caches() -> None:
    """Clear Streamlit resource caches before each UI test."""
    st.cache_resource.clear()


def _make_settings(sample_csv_file: Path, tmp_path: Path, tag: str) -> AppSettings:
    return AppSettings(
        gemini_api_key="test-key",
        gemini_model="gemini-2.0-flash",
        db_path=tmp_path / f"{tag}.db",
        csv_path=sample_csv_file,
        chroma_path=tmp_path / f"chroma_{tag}",
    )


def test_app_stops_gracefully_when_api_key_missing() -> None:
    with patch(
        "config.load_settings",
        side_effect=ConfigurationError("GEMINI_API_KEY tidak ditemukan."),
    ):
        at = AppTest.from_file("app.py", default_timeout=APP_TEST_TIMEOUT).run()

    assert len(at.error) == 1
    assert "GEMINI_API_KEY" in at.error[0].value


def test_app_renders_tanya_gizi_with_sqlite_fallback_when_chroma_empty(
    sample_csv_file: Path,
    tmp_path: Path,
) -> None:
    fake_settings = _make_settings(sample_csv_file, tmp_path, "fallback")
    fake_client = FakeChromaClient()
    fake_model = FakeGeminiModel(reply="Nasi goreng 250 kal.")

    with (
        patch("config.load_settings", return_value=fake_settings),
        patch("services.llm.create_gemini_model", return_value=fake_model),
        patch("services.embeddings.get_chroma_client", return_value=fake_client),
        patch("services.embeddings.create_gemini_embedding_function", return_value=object()),
    ):
        at = AppTest.from_file("app.py", default_timeout=APP_TEST_TIMEOUT).run()
        assert len(at.sidebar.warning) == 1

        at.text_input(key="rag_query_input").input("Nasi Goreng").run()
        at.button(key="rag_query_button").click().run()

        assert not at.exception
        assert len(at.main.warning) == 1
        assert len(at.dataframe) == 1


def test_app_all_menus_happy_path_and_indexing(
    sample_csv_file: Path,
    tmp_path: Path,
) -> None:
    fake_settings = _make_settings(sample_csv_file, tmp_path, "menus")
    fake_client = FakeChromaClient()
    fake_model = FakeGeminiModel(reply="Output AI Valid")

    with (
        patch("config.load_settings", return_value=fake_settings),
        patch("services.llm.create_gemini_model", return_value=fake_model),
        patch("services.embeddings.get_chroma_client", return_value=fake_client),
        patch("services.embeddings.create_gemini_embedding_function", return_value=object()),
    ):
        at = AppTest.from_file("app.py", default_timeout=APP_TEST_TIMEOUT).run()

        # 1. Masak Apa Hari Ini?
        at.sidebar.radio[0].set_value("Masak Apa Hari Ini?").run()
        at.text_input(key="recipe_ingredients").input("ayam, brokoli").run()
        at.button(key="get_recipe_button").click().run()
        assert not at.exception
        assert any("Output AI Valid" in md.value for md in at.markdown)

        # 2. Perencana Menu Mingguan
        at.sidebar.radio[0].set_value("Perencana Menu Mingguan").run()
        at.button(key="generate_menu_button").click().run()
        assert not at.exception
        assert any("Output AI Valid" in md.value for md in at.markdown)

        # 3. Alternatif Sehat
        at.sidebar.radio[0].set_value("Alternatif Sehat").run()
        at.text_input(key="healthy_alternative_input").input("gula").run()
        at.button(key="find_alternative_button").click().run()
        assert not at.exception
        assert any("Output AI Valid" in md.value for md in at.markdown)

        # 4. Index Data
        at.sidebar.radio[0].set_value("Index Data").run()
        at.button(key="reindex_button").click().run()
        assert not at.exception
        assert len(at.success) == 1
        assert "3 item ke ChromaDB" in at.success[0].value

        # 5. Tanya Gizi after indexing now succeeds via RAG
        at.sidebar.radio[0].set_value("Tanya Gizi").run()
        at.text_input(key="rag_query_input").input("Berapa kalori nasi goreng?").run()
        at.button(key="rag_query_button").click().run()
        assert not at.exception
        assert any("Output AI Valid" in md.value for md in at.markdown)
