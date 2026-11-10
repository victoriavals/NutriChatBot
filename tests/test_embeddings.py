"""Unit tests for ChromaDB vector indexing and RAG retrieval service."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from services.embeddings import (
    EmbeddingServiceError,
    create_gemini_embedding_function,
    get_chroma_client,
    get_collection_count,
    index_nutrition_data,
    retrieve_nutrition_documents,
    sanitize_metadata_records,
)
from tests.conftest import FakeChromaClient


def test_importing_embeddings_has_no_api_key_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    mod = importlib.import_module("embeddings")
    importlib.reload(mod)
    assert hasattr(mod, "rag_query")


def test_create_gemini_embedding_function_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_gemini_embedding_function("   ")


def test_sanitize_metadata_records_handles_nan_inf_and_numpy() -> None:
    df = pd.DataFrame(
        [
            {
                "name": "Tempe",
                "calories": np.float64(193.0),
                "image": float("nan"),
                "extra_inf": math.inf,
                "description": "Tempe desc",
            }
        ]
    )
    records = sanitize_metadata_records(df)
    assert len(records) == 1
    assert "description" not in records[0]
    assert records[0]["name"] == "Tempe"
    assert isinstance(records[0]["calories"], float)
    assert records[0]["image"] == 0.0
    assert records[0]["extra_inf"] == 0.0


def test_index_nutrition_data_batches_and_replaces_existing() -> None:
    client = FakeChromaClient()
    df = pd.DataFrame(
        [
            {"name": f"Food {i}", "calories": float(i * 10), "description": f"Food {i} desc"}
            for i in range(5)
        ]
    )
    ef: Any = object()

    # First index
    count = index_nutrition_data(df, client=client, embedding_function=ef, batch_size=2)
    assert count == 5
    collection = client.get_collection("nutrition")
    assert len(collection.batches) == 3  # 2 + 2 + 1
    assert get_collection_count(client, ef) == 5

    # Re-index deletes previous collection cleanly
    count_second = index_nutrition_data(df, client=client, embedding_function=ef, batch_size=3)
    assert count_second == 5
    assert "nutrition" in client.deleted_names


def test_retrieve_nutrition_documents_empty_or_missing_raises() -> None:
    client = FakeChromaClient()
    ef: Any = object()

    with pytest.raises(EmbeddingServiceError, match="belum tersedia"):
        retrieve_nutrition_documents("nasi goreng", client=client, embedding_function=ef)

    # Empty collection also raises helpful error
    client.get_or_create_collection("nutrition", ef)
    with pytest.raises(EmbeddingServiceError, match="kosong"):
        retrieve_nutrition_documents("nasi goreng", client=client, embedding_function=ef)


def test_retrieve_nutrition_documents_caps_n_results() -> None:
    client = FakeChromaClient()
    ef: Any = object()
    col = client.get_or_create_collection("nutrition", ef)
    col.add(
        documents=["Nasi Goreng 250 cal", "Mie Goreng 300 cal"],
        metadatas=[{"name": "Nasi Goreng"}, {"name": "Mie Goreng"}],
        ids=["doc_0", "doc_1"],
    )

    docs = retrieve_nutrition_documents(
        "goreng",
        client=client,
        embedding_function=ef,
        n_results=5,
    )
    assert docs == ["Nasi Goreng 250 cal", "Mie Goreng 300 cal"]


def test_real_chromadb_persistent_client_roundtrip(tmp_path: Path) -> None:
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class DeterministicTestEF(EmbeddingFunction[Documents]):
        def __init__(self) -> None:
            pass

        def __call__(self, input: Documents) -> Embeddings:
            vectors: list[list[float]] = []
            for text in input:
                lowered = text.lower()
                vectors.append(
                    [
                        1.0 if "nasi" in lowered else 0.0,
                        1.0 if "ayam" in lowered else 0.0,
                        float(len(lowered) % 10) / 10.0,
                    ]
                )
            return vectors  # type: ignore[return-value]

        @staticmethod
        def name() -> str:
            return "deterministic-test-ef"

        def get_config(self) -> dict[str, Any]:
            return {}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> DeterministicTestEF:
            return DeterministicTestEF()

    ef = DeterministicTestEF()
    client = get_chroma_client(tmp_path / "real_chroma")
    assert get_collection_count(client, ef) == 0

    df = pd.DataFrame(
        [
            {
                "name": "Nasi Goreng",
                "calories": 250.0,
                "image": float("nan"),
                "description": "Nasi Goreng has 250.0 calories",
            },
            {
                "name": "Dada Ayam",
                "calories": 165.0,
                "image": None,
                "description": "Dada Ayam has 165.0 calories",
            },
        ]
    )

    indexed = index_nutrition_data(df, client=client, embedding_function=ef, batch_size=1)
    assert indexed == 2
    assert get_collection_count(client, ef) == 2

    retrieved = retrieve_nutrition_documents("nasi", client=client, embedding_function=ef, n_results=1)
    assert retrieved == ["Nasi Goreng has 250.0 calories"]


