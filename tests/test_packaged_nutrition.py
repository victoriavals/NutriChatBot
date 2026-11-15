"""Unit tests for packaged FMCG food database operations."""

from pathlib import Path

import pandas as pd
import pytest

from services.database import (
    ensure_packaged_nutrition_initialized,
    get_packaged_categories,
    get_packaged_nutrition_count,
    save_packaged_nutrition_to_sqlite,
    search_packaged_foods,
)


@pytest.fixture
def sample_packaged_df() -> pd.DataFrame:
    data = [
        {
            "id": 1,
            "name": "Indomie Mi Goreng Spesial",
            "brand": "Indofood",
            "category": "Mie Instan",
            "serving_size_g": 85.0,
            "serving_unit": "1 bungkus (85g)",
            "calories": 380.0,
            "proteins": 8.0,
            "fat": 14.0,
            "carbohydrate": 54.0,
            "saturated_fat": 6.0,
            "sugars": 8.0,
            "sodium_mg": 1070.0,
            "nutri_score": "E",
            "description": "Mie instan goreng populer.",
        },
        {
            "id": 2,
            "name": "Ultra Milk UHT Low Fat 250ml",
            "brand": "Ultrajaya",
            "category": "Susu & Olahan",
            "serving_size_g": 250.0,
            "serving_unit": "1 kotak (250ml)",
            "calories": 110.0,
            "proteins": 8.0,
            "fat": 3.0,
            "carbohydrate": 13.0,
            "saturated_fat": 1.5,
            "sugars": 10.0,
            "sodium_mg": 105.0,
            "nutri_score": "A",
            "description": "Susu rendah lemak tinggi kalsium.",
        },
        {
            "id": 3,
            "name": "Quaker Instant Oatmeal Merah",
            "brand": "Quaker",
            "category": "Sereal & Oat",
            "serving_size_g": 35.0,
            "serving_unit": "4 sdm (35g)",
            "calories": 130.0,
            "proteins": 4.0,
            "fat": 2.5,
            "carbohydrate": 23.0,
            "saturated_fat": 0.5,
            "sugars": 0.5,
            "sodium_mg": 5.0,
            "nutri_score": "A",
            "description": "Oatmeal instan gandum utuh.",
        },
    ]
    return pd.DataFrame(data)


def test_save_and_count_packaged_nutrition(
    sample_packaged_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "packaged_test.db"
    assert get_packaged_nutrition_count(db_path) == 0

    rows_written = save_packaged_nutrition_to_sqlite(sample_packaged_df, db_path)
    assert rows_written == 3
    assert get_packaged_nutrition_count(db_path) == 3


def test_get_packaged_categories(
    sample_packaged_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "packaged_cat.db"
    save_packaged_nutrition_to_sqlite(sample_packaged_df, db_path)

    categories = get_packaged_categories(db_path)
    assert "Mie Instan" in categories
    assert "Susu & Olahan" in categories
    assert "Sereal & Oat" in categories


def test_search_packaged_foods_by_brand_and_nutri_score(
    sample_packaged_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "packaged_search.db"
    save_packaged_nutrition_to_sqlite(sample_packaged_df, db_path)

    # Search by brand
    indofood_results = search_packaged_foods(query="Indofood", db_path=db_path)
    assert len(indofood_results) == 1
    assert indofood_results[0]["name"] == "Indomie Mi Goreng Spesial"

    # Search with Nutri-Score filter A
    grade_a_results = search_packaged_foods(nutri_score="A", db_path=db_path)
    assert len(grade_a_results) == 2
    names = [r["name"] for r in grade_a_results]
    assert "Ultra Milk UHT Low Fat 250ml" in names
    assert "Quaker Instant Oatmeal Merah" in names


def test_ensure_packaged_nutrition_initialized_from_csv(
    sample_packaged_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    csv_file = tmp_path / "packaged_sample.csv"
    sample_packaged_df.to_csv(csv_file, index=False)
    db_file = tmp_path / "auto_packaged.db"

    assert ensure_packaged_nutrition_initialized(csv_file, db_file) is True
    assert get_packaged_nutrition_count(db_file) == 3
    assert ensure_packaged_nutrition_initialized(csv_file, db_file) is False
