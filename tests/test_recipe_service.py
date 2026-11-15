"""Unit tests for authentic Indonesian recipe database and recommendation service."""

import json
from pathlib import Path

import pandas as pd
import pytest

from services.database import (
    ensure_recipes_initialized,
    get_recipe_by_id,
    get_recipe_categories,
    get_recipes_count,
    save_recipes_to_sqlite,
    search_recipes,
)
from services.recipe_service import (
    build_recipe_adaptation_prompt,
    find_recipes_by_pantry_ingredients,
)


@pytest.fixture
def sample_recipes_df() -> pd.DataFrame:
    data = [
        {
            "id": 1,
            "name": "Soto Ayam Lamongan",
            "origin_region": "Jawa Timur",
            "category": "Sup & Soto",
            "prep_time_minutes": 35,
            "servings": 4,
            "ingredients_json": json.dumps(["dada ayam", "bihun", "telur", "tauge", "serai"]),
            "instructions_text": "Rebus ayam dan racik bumbu soto.",
            "calories_per_serving": 240.0,
            "protein_g": 26.0,
            "fat_g": 5.0,
            "carb_g": 22.0,
            "health_tags": "Tinggi Protein, Rendah Lemak",
        },
        {
            "id": 2,
            "name": "Tumis Kangkung Terasi",
            "origin_region": "Nusantara",
            "category": "Tumis & Sayur",
            "prep_time_minutes": 15,
            "servings": 3,
            "ingredients_json": json.dumps(["kangkung", "bawang putih", "terasi"]),
            "instructions_text": "Tumis bawang dan kangkung cepat.",
            "calories_per_serving": 85.0,
            "protein_g": 3.0,
            "fat_g": 4.5,
            "carb_g": 9.0,
            "health_tags": "Tinggi Serat, Cepat Saji",
        },
        {
            "id": 3,
            "name": "Pepes Tahu Jamur",
            "origin_region": "Sunda",
            "category": "Kukus & Pepes",
            "prep_time_minutes": 30,
            "servings": 4,
            "ingredients_json": json.dumps(["tahu putih", "jamur tiram", "kemangi", "telur"]),
            "instructions_text": "Bungkus daun pisang lalu kukus.",
            "calories_per_serving": 125.0,
            "protein_g": 10.0,
            "fat_g": 4.5,
            "carb_g": 11.0,
            "health_tags": "Vegetarian Sehat, Rendah Lemak",
        },
    ]
    return pd.DataFrame(data)


def test_save_and_retrieve_recipes(
    sample_recipes_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recipes_test.db"
    assert get_recipes_count(db_path) == 0

    rows = save_recipes_to_sqlite(sample_recipes_df, db_path)
    assert rows == 3
    assert get_recipes_count(db_path) == 3

    categories = get_recipe_categories(db_path)
    assert "Sup & Soto" in categories
    assert "Tumis & Sayur" in categories


def test_search_recipes_by_health_tag_and_category(
    sample_recipes_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recipes_search.db"
    save_recipes_to_sqlite(sample_recipes_df, db_path)

    # Search with health tag
    protein_recipes = search_recipes(health_tag="Tinggi Protein", db_path=db_path)
    assert len(protein_recipes) == 1
    assert protein_recipes[0]["name"] == "Soto Ayam Lamongan"

    # Search by ID
    soto = get_recipe_by_id(1, db_path=db_path)
    assert soto is not None
    assert soto["name"] == "Soto Ayam Lamongan"


def test_find_recipes_by_pantry_ingredients_ranks_matches(
    sample_recipes_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recipes_match.db"
    save_recipes_to_sqlite(sample_recipes_df, db_path)

    # User has ayam and telur
    matches = find_recipes_by_pantry_ingredients(
        available_ingredients=["ayam", "telur"],
        db_path=db_path,
    )
    assert len(matches) >= 1
    # Soto Ayam Lamongan matches 2 ingredients ('dada ayam', 'telur')
    top_match = matches[0]
    assert top_match.name == "Soto Ayam Lamongan"
    assert "dada ayam" in top_match.matched_ingredients
    assert "telur" in top_match.matched_ingredients
    assert top_match.match_percentage > 0.0


def test_build_recipe_adaptation_prompt_includes_details(
    sample_recipes_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "recipes_prompt.db"
    save_recipes_to_sqlite(sample_recipes_df, db_path)

    prompt = build_recipe_adaptation_prompt(
        recipe_id=1,
        user_notes="Untuk penderita hipertensi, kurangi garam",
        db_path=db_path,
    )
    assert "Soto Ayam Lamongan" in prompt
    assert "hipertensi" in prompt
    assert "240.0 kkal/porsi" in prompt


def test_ensure_recipes_initialized_from_csv(
    sample_recipes_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    csv_file = tmp_path / "recipes_sample.csv"
    sample_recipes_df.to_csv(csv_file, index=False)
    db_file = tmp_path / "auto_recipes.db"

    assert ensure_recipes_initialized(csv_file, db_file) is True
    assert get_recipes_count(db_file) == 3
    assert ensure_recipes_initialized(csv_file, db_file) is False
