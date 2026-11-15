"""Indonesian household serving size (Ukuran Rumah Tangga / URT) definitions and converter.

Translates clinical 100-gram standard food composition values into practical,
everyday Indonesian serving portions (e.g. centong, potong, butir, mangkok, sdm).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ServingOption:
    """Represents an Indonesian household serving measurement option."""

    unit_id: str
    label: str
    grams: float
    description: str


# Standard baseline fallback available for all food items
DEFAULT_SERVING_OPTIONS: tuple[ServingOption, ...] = (
    ServingOption("porsi_100g", "100 gram (Porsi Standar DKBM)", 100.0, "Takaran standar laboratorium"),
    ServingOption("porsi_50g", "Setengah Porsi (~50g)", 50.0, "Porsi kecil / cicip"),
    ServingOption("porsi_200g", "Porsi Ganda (~200g)", 200.0, "Porsi besar / kenyang"),
    ServingOption("sendok_makan", "1 Sendok Makan (~15g)", 15.0, "Takaran sendok makan makan"),
)

# Domain-specific Indonesian dietary URT mapping based on food taxonomy patterns
_URT_CATEGORY_MAP: tuple[tuple[re.Pattern[str], tuple[ServingOption, ...]], ...] = (
    # Nasi / Beras / Bubur / Ketan
    (
        re.compile(r"\b(nasi|beras|bubur|ketan|lontong|ketupat)\b", re.IGNORECASE),
        (
            ServingOption("centong", "1 Centong Rice Cooker (~100g)", 100.0, "Porsi lazim 1 centong nasi"),
            ServingOption("piring_sedang", "1 Piring Sedang (~200g)", 200.0, "Porsi makan standar rumahan"),
            ServingOption("centong_dua", "2 Centong Rice Cooker (~200g)", 200.0, "Porsi makan kenyang pria/pekerja"),
            ServingOption("mangkok", "1 Mangkok Sedang (~150g)", 150.0, "Porsi mangkok sup/resto"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Telur
    (
        re.compile(r"\b(telur|telor)\b", re.IGNORECASE),
        (
            ServingOption("butir_utuh", "1 Butir Utuh (~55g)", 55.0, "1 butir telur ayam ras ukuran sedang"),
            ServingOption("dua_butir", "2 Butir Utuh (~110g)", 110.0, "2 butir telur sarapan / olahan"),
            ServingOption("putih_telur", "1 Butir Putih Telur (~35g)", 35.0, "Hanya bagian putih untuk diet protein"),
            ServingOption("kuning_telur", "1 Butir Kuning Telur (~20g)", 20.0, "Hanya bagian kuning"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Setara ~2 butir telur sedang"),
        ),
    ),
    # Daging Ayam / Bebek
    (
        re.compile(r"\b(ayam|bebek|unggas)\b", re.IGNORECASE),
        (
            ServingOption("potong_sedang", "1 Potong Sedang (~80g BDD)", 80.0, "1 potong paha/dada tanpa tulang"),
            ServingOption("potong_besar", "1 Potong Besar (~120g BDD)", 120.0, "1 potong dada utuh / paha atas"),
            ServingOption("potong_kecil", "1 Potong Kecil / Suwir (~40g)", 40.0, "Topping soto / bubur / mie"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Daging Sapi / Kambing / Kerbau
    (
        re.compile(r"\b(sapi|daging|kambing|kornet)\b", re.IGNORECASE),
        (
            ServingOption("potong_sedang", "1 Potong Sedang Rendang/Empal (~50g)", 50.0, "1 potong olahan masakan"),
            ServingOption("dua_potong", "2 Potong Sedang (~100g)", 100.0, "2 potong rendang / steak porsi saji"),
            ServingOption("sendok_makan", "1 Sendok Makan Cincang (~20g)", 20.0, "Campuran tumisan / sup"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Ikan / Udang / Cumi / Seafood
    (
        re.compile(r"\b(ikan|udang|cumi|kepiting|tongkol|teri|lele|gurame|nila|kakap)\b", re.IGNORECASE),
        (
            ServingOption("ekor_sedang", "1 Ekor Sedang (Lele/Kembung ~80g BDD)", 80.0, "1 ekor bersih porsi saji"),
            ServingOption("potong_sedang", "1 Potong Sedang Gurame/Tongkol (~75g)", 75.0, "1 potong bagian tengah"),
            ServingOption("porsi_udang", "5 Ekor Udang Sedang (~50g)", 50.0, "Porsi lauk udang matang"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Tempe / Tahu / Oncom
    (
        re.compile(r"\b(tempe|tahu|oncom)\b", re.IGNORECASE),
        (
            ServingOption("potong_sedang", "1 Potong Sedang (~50g)", 50.0, "Ukuran 5x4x1 cm potongan lauk"),
            ServingOption("dua_potong", "2 Potong Sedang (~100g)", 100.0, "2 potong tempe/tahu goreng"),
            ServingOption("buah_tahu", "1 Buah Tahu Kotak Segar (~70g)", 70.0, "1 blok tahu cina / tahu putih"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Sayuran Hijau & Berkuah (Bayam, Kangkung, Sawi, Sayur Asem, Sop)
    (
        re.compile(r"\b(bayam|kangkung|sawi|kol|kubis|sop|sayur|asem|lodeh|wortel|buncis)\b", re.IGNORECASE),
        (
            ServingOption("mangkok", "1 Mangkok Sayur Matang (~100g BDD)", 100.0, "Porsi 1 mangkok sup sayur segar"),
            ServingOption("centong_sayur", "1 Centong Sayur Berkuah (~50g)", 50.0, "1 centong kuah dan sayuran"),
            ServingOption("piring", "1 Piring Penuh Tumisan (~150g)", 150.0, "Porsi makan berdua / porsi besar"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Buah-buahan
    (
        re.compile(r"\b(pisang|apel|jeruk|mangga|pepaya|semangka|melon|alpukat)\b", re.IGNORECASE),
        (
            ServingOption("buah_sedang", "1 Buah Sedang (~100g BDD)", 100.0, "1 buah apel/pisang/jeruk utuh"),
            ServingOption("potong_besar", "1 Potong Irisan Besar Semangka/Pepaya (~120g)", 120.0, "1 irisan memanjang"),
            ServingOption("setengah_buah", "Setengah Buah Alpukat/Mangga (~75g)", 75.0, "Setengah porsi santap"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Minuman / Cairan / Susu / Minyak / Sirup
    (
        re.compile(r"\b(susu|minyak|mentega|margarin|santan|kecap|sirup|teh|kopi|jus)\b", re.IGNORECASE),
        (
            ServingOption("gelas", "1 Gelas Belimbing (~250 ml / 250g)", 250.0, "Porsi 1 gelas standar"),
            ServingOption("cangkir", "1 Cangkir Teh/Kopi (~150 ml / 150g)", 150.0, "Porsi cangkir santap"),
            ServingOption("sendok_makan", "1 Sendok Makan (~15g / 15ml)", 15.0, "Takaran minyak goreng / bumbu"),
            ServingOption("sendok_teh", "1 Sendok Teh (~5g / 5ml)", 5.0, "Takaran gula / minyak sedikit"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
    # Mie / Bihun / Pasta / Roti
    (
        re.compile(r"\b(mie|mi|bihun|kwetiau|roti|pasta|makaroni)\b", re.IGNORECASE),
        (
            ServingOption("porsi_saji", "1 Porsi Mangkuk / Piring Matang (~150g)", 150.0, "1 porsi mie ayam / bakmi"),
            ServingOption("lembar_roti", "1 Lembar Roti Tawar (~35g)", 35.0, "1 lembar roti tawar kupas"),
            ServingOption("dua_lembar", "2 Lembar Roti Tawar (~70g)", 70.0, "Porsi sandwich sarapan"),
            ServingOption("porsi_100g", "100 gram (Porsi Standar)", 100.0, "Standar acuan DKBM"),
        ),
    ),
)


def get_serving_options(food_name: str) -> list[ServingOption]:
    """Retrieve appropriate Indonesian household serving options for a food item.

    Matches food names against Indonesian nutritional taxonomy rules and returns
    human-intuitive units (centong, butir, potong, etc.). Falls back to standard
    percentage fractions of 100g if no specific pattern matches.

    Args:
        food_name: Common name of the food item (e.g. 'Nasi putih', 'Telur ayam').

    Returns:
        List of ServingOption objects.
    """
    cleaned_name = food_name.strip()
    if not cleaned_name:
        return list(DEFAULT_SERVING_OPTIONS)

    for pattern, options in _URT_CATEGORY_MAP:
        if pattern.search(cleaned_name):
            return list(options)

    return list(DEFAULT_SERVING_OPTIONS)


def calculate_nutrition_for_serving(
    calories_100g: float,
    proteins_100g: float,
    fat_100g: float,
    carb_100g: float,
    serving_grams: float,
) -> dict[str, float]:
    """Calculate scaled macronutrients for an Indonesian household serving weight.

    Args:
        calories_100g: Energy in kcal per 100 grams.
        proteins_100g: Protein in grams per 100 grams.
        fat_100g: Fat in grams per 100 grams.
        carb_100g: Carbohydrates in grams per 100 grams.
        serving_grams: Selected serving weight in grams (must be > 0).

    Returns:
        Dictionary with scaled 'calories', 'proteins', 'fat', 'carbohydrate',
        and 'serving_grams', all rounded to 1 decimal place.

    Raises:
        ValueError: If serving_grams is non-positive or nutrients are negative.
    """
    if serving_grams <= 0:
        raise ValueError(f"Berat porsi harus lebih besar dari 0 gram (diberikan: {serving_grams}).")

    for name, val in [
        ("kalori", calories_100g),
        ("protein", proteins_100g),
        ("lemak", fat_100g),
        ("karbohidrat", carb_100g),
    ]:
        if val < 0:
            raise ValueError(f"Nilai {name} tidak boleh negatif (diberikan: {val}).")

    ratio = serving_grams / 100.0
    return {
        "serving_grams": round(serving_grams, 1),
        "calories": round(calories_100g * ratio, 1),
        "proteins": round(proteins_100g * ratio, 1),
        "fat": round(fat_100g * ratio, 1),
        "carbohydrate": round(carb_100g * ratio, 1),
    }


def format_serving_summary(
    food_name: str,
    serving_label: str,
    nutrition_dict: dict[str, Any],
) -> str:
    """Format a layperson-friendly summary of nutrition for a specific serving size.

    Args:
        food_name: Food item name.
        serving_label: Label of the serving portion (e.g. '1 Centong (~100g)').
        nutrition_dict: Dictionary with scaled macronutrient values.

    Returns:
        Formatted human-readable string.
    """
    return (
        f"**{food_name}** porsi **{serving_label}** mengandung sekitar "
        f"**{nutrition_dict.get('calories', 0.0):.1f} kkal** "
        f"(Protein: {nutrition_dict.get('proteins', 0.0):.1f}g, "
        f"Lemak: {nutrition_dict.get('fat', 0.0):.1f}g, "
        f"Karbohidrat: {nutrition_dict.get('carbohydrate', 0.0):.1f}g)."
    )
