"""Recipe recommendation and ingredient matching service for authentic Indonesian dishes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config import DEFAULT_DB_PATH, get_logger
from services.database import get_recipe_by_id, search_recipes

logger = get_logger(__name__)


@dataclass(frozen=True)
class RecipeMatch:
    """Represents an authentic recipe matched against user's pantry ingredients."""

    recipe_id: int
    name: str
    origin_region: str
    category: str
    prep_time_minutes: int
    servings: int
    calories_per_serving: float
    protein_g: float
    fat_g: float
    carb_g: float
    health_tags: str
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    match_percentage: float
    instructions_text: str


def find_recipes_by_pantry_ingredients(
    available_ingredients: list[str],
    health_filter: str = "",
    limit: int = 6,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[RecipeMatch]:
    """Find authentic recipes matching user's pantry ingredients and calculate match scores.

    Args:
        available_ingredients: List of ingredients user has (e.g. ['ayam', 'tahu', 'bayam']).
        health_filter: Optional health condition filter (e.g. 'Rendah Lemak', 'Tinggi Protein').
        limit: Maximum number of recipe matches to return.
        db_path: Path to SQLite database.

    Returns:
        List of RecipeMatch sorted by match percentage descending.
    """
    cleaned_inputs = [i.strip().lower() for i in available_ingredients if i.strip()]
    all_recipes = search_recipes(
        query="",
        category="",
        health_tag=health_filter,
        limit=50,
        db_path=db_path,
    )
    if not all_recipes:
        return []

    # If user provided no ingredients, return all recipes with 100% match
    if not cleaned_inputs:
        matches = []
        for r in all_recipes[:limit]:
            try:
                ing_list = json.loads(r["ingredients_json"])
            except (json.JSONDecodeError, TypeError, KeyError):
                ing_list = []
            matches.append(
                RecipeMatch(
                    recipe_id=r["id"],
                    name=r["name"],
                    origin_region=r["origin_region"],
                    category=r["category"],
                    prep_time_minutes=r["prep_time_minutes"],
                    servings=r["servings"],
                    calories_per_serving=r["calories_per_serving"],
                    protein_g=r["protein_g"],
                    fat_g=r["fat_g"],
                    carb_g=r["carb_g"],
                    health_tags=r["health_tags"],
                    matched_ingredients=ing_list,
                    missing_ingredients=[],
                    match_percentage=100.0,
                    instructions_text=r["instructions_text"],
                )
            )
        return matches

    scored_matches: list[RecipeMatch] = []
    for r in all_recipes:
        try:
            ing_list = json.loads(r["ingredients_json"])
        except (json.JSONDecodeError, TypeError, KeyError):
            ing_list = []

        matched_ings: list[str] = []
        missing_ings: list[str] = []

        for ing in ing_list:
            ing_lower = ing.lower()
            if any(user_item in ing_lower for user_item in cleaned_inputs):
                matched_ings.append(ing)
            else:
                missing_ings.append(ing)

        # Only consider recipes that match at least one user ingredient
        if matched_ings:
            match_pct = (len(matched_ings) / max(len(ing_list), 1)) * 100.0
            scored_matches.append(
                RecipeMatch(
                    recipe_id=r["id"],
                    name=r["name"],
                    origin_region=r["origin_region"],
                    category=r["category"],
                    prep_time_minutes=r["prep_time_minutes"],
                    servings=r["servings"],
                    calories_per_serving=r["calories_per_serving"],
                    protein_g=r["protein_g"],
                    fat_g=r["fat_g"],
                    carb_g=r["carb_g"],
                    health_tags=r["health_tags"],
                    matched_ingredients=matched_ings,
                    missing_ingredients=missing_ings,
                    match_percentage=round(match_pct, 1),
                    instructions_text=r["instructions_text"],
                )
            )

    scored_matches.sort(
        key=lambda m: (m.match_percentage, -m.prep_time_minutes),
        reverse=True,
    )
    return scored_matches[:limit]


def build_recipe_adaptation_prompt(
    recipe_id: int,
    user_notes: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> str:
    """Build a prompt for AI to modify or adapt an authentic regional recipe."""
    recipe = get_recipe_by_id(recipe_id, db_path=db_path)
    if not recipe:
        return f"Buatkan panduan memasak sehat untuk resep nusantara dengan catatan: {user_notes}"

    try:
        ing_list = json.loads(recipe["ingredients_json"])
        formatted_ingredients = "\n".join(f"- {ing}" for ing in ing_list)
    except (json.JSONDecodeError, TypeError, KeyError):
        formatted_ingredients = str(recipe.get("ingredients_json", ""))

    return (
        f"Sebagai Chef Spesialis Masakan Sehat Nusantara, berikan panduan modifikasi untuk resep otentik berikut:\n\n"
        f"**Nama Resep:** {recipe['name']} (Asal: {recipe['origin_region']})\n"
        f"**Kategori:** {recipe['category']} | Waktu: {recipe['prep_time_minutes']} menit | Porsi: {recipe['servings']} orang\n"
        f"**Nilai Gizi Asli:** {recipe['calories_per_serving']} kkal/porsi "
        f"(Protein: {recipe['protein_g']}g, Lemak: {recipe['fat_g']}g, Karbohidrat: {recipe['carb_g']}g)\n"
        f"**Bahan Asli:**\n{formatted_ingredients}\n\n"
        f"**Langkah Asli:**\n{recipe['instructions_text']}\n\n"
        f"**Catatan/Kebutuhan Diet Pengguna:** {user_notes or 'Jadikan lebih sehat, rendah minyak, dan kaya gizi.'}\n\n"
        f"Jelaskan: 1) Bahan pengganti yang lebih sehat (jika ada), 2) Tips memasak agar nutrisi tidak rusak, "
        f"dan 3) Estimasi perubahan kalori jika mengikuti saran modifikasi Anda."
    )
