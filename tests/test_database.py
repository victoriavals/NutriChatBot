"""Unit and integration tests for SQLite database and CSV validation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from services.database import (
    DatabaseError,
    detect_nutrition_anomalies,
    ensure_sqlite_initialized,
    get_db_connection,
    get_nutrition_count,
    load_and_validate_csv,
    save_nutrition_to_sqlite,
    search_nutrition,
)


def test_load_and_validate_csv_builds_descriptions(sample_csv_file: Path) -> None:
    df, anomalies = load_and_validate_csv(sample_csv_file)
    assert len(df) == 3
    assert "description" in df.columns
    assert "Nasi Goreng has 250.0 calories" in df.iloc[0]["description"]
    assert df.iloc[1]["image"] == ""
    assert anomalies == []


def test_load_and_validate_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"name": "Apel", "calories": 52}]).to_csv(bad_csv, index=False)

    with pytest.raises(DatabaseError, match="proteins"):
        load_and_validate_csv(bad_csv)


def test_load_and_validate_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatabaseError, match="tidak ditemukan"):
        load_and_validate_csv(tmp_path / "does_not_exist.csv")


def test_save_and_search_nutrition_escapes_like_wildcards(
    sample_csv_file: Path,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "test.db"
    df, _ = load_and_validate_csv(sample_csv_file)
    written = save_nutrition_to_sqlite(df, db_path)

    assert written == 3
    assert get_nutrition_count(db_path) == 3

    # Literal '%' wildcard should only match the item containing '100%', not all rows
    percent_matches = search_nutrition("100%", db_path=db_path)
    assert len(percent_matches) == 1
    assert percent_matches[0]["name"] == "100% Tempe_Goreng"

    # Case-insensitive search
    chicken_matches = search_nutrition("dada ayam", db_path=db_path)
    assert len(chicken_matches) == 1
    assert chicken_matches[0]["proteins"] == 31.0

    # Empty query returns empty list
    assert search_nutrition("   ", db_path=db_path) == []


def test_ensure_sqlite_initialized_is_idempotent(
    sample_csv_file: Path,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "auto.db"
    assert ensure_sqlite_initialized(sample_csv_file, db_path) is True
    assert get_nutrition_count(db_path) == 3
    assert ensure_sqlite_initialized(sample_csv_file, db_path) is False


def test_get_db_connection_closes_even_on_exception(tmp_path: Path) -> None:
    db_path = tmp_path / "close_test.db"
    captured_conn: sqlite3.Connection | None = None

    with pytest.raises(RuntimeError), get_db_connection(db_path) as conn:
        captured_conn = conn
        raise RuntimeError("boom")

    assert captured_conn is not None
    with pytest.raises(sqlite3.ProgrammingError):
        captured_conn.execute("SELECT 1")


def test_detect_nutrition_anomalies_flags_implausible_macros() -> None:
    df = pd.DataFrame(
        [
            {"name": "Pilus", "calories": 514.0, "proteins": 5.0, "fat": 10.0, "carbohydrate": 647.0},
            {"name": "Agar-agar", "calories": 0.0, "proteins": 0.0, "fat": 0.2, "carbohydrate": 0.0},
            {"name": "Invalid", "calories": -10.0, "proteins": 1.0, "fat": 1.0, "carbohydrate": 1.0},
        ]
    )
    warnings = detect_nutrition_anomalies(df)
    assert len(warnings) == 3

