"""Unit tests for Indonesian Household Serving Size (URT) service."""

import pytest

from services.serving_size import (
    DEFAULT_SERVING_OPTIONS,
    calculate_nutrition_for_serving,
    format_serving_summary,
    get_serving_options,
)


def test_get_serving_options_matches_nasi_patterns():
    """Verify that rice/nasi keywords resolve to centong and piring options."""
    options = get_serving_options("Nasi putih giling")
    unit_ids = [opt.unit_id for opt in options]

    assert "centong" in unit_ids
    assert "piring_sedang" in unit_ids
    assert any(opt.grams == 100.0 for opt in options)


def test_get_serving_options_matches_telur_patterns():
    """Verify that egg items resolve to butir and white/yolk options."""
    options = get_serving_options("Telur ayam kampung")
    unit_ids = [opt.unit_id for opt in options]

    assert "butir_utuh" in unit_ids
    assert "putih_telur" in unit_ids
    assert any(opt.grams == 55.0 for opt in options)


def test_get_serving_options_matches_ayam_and_daging():
    """Verify poultry and meat resolve to potong sedang/besar options."""
    ayam_options = get_serving_options("Dada ayam goreng")
    assert any("potong_sedang" in opt.unit_id for opt in ayam_options)

    daging_options = get_serving_options("Rendang daging sapi")
    assert any("potong_sedang" in opt.unit_id for opt in daging_options)


def test_get_serving_options_fallback_on_unknown_or_empty():
    """Verify unknown food names or empty strings gracefully return default options."""
    options_unknown = get_serving_options("Kue alien dari mars")
    assert options_unknown == list(DEFAULT_SERVING_OPTIONS)

    options_empty = get_serving_options("")
    assert options_empty == list(DEFAULT_SERVING_OPTIONS)


def test_calculate_nutrition_for_serving_scales_accurately():
    """Verify proportional calculation of calories and macros for household grams."""
    # 100g nasi: 180 kcal, 3.0g protein, 0.3g fat, 39.8g carb
    # Scaled to 1 piring sedang (200g) -> ratio 2.0
    scaled = calculate_nutrition_for_serving(
        calories_100g=180.0,
        proteins_100g=3.0,
        fat_100g=0.3,
        carb_100g=39.8,
        serving_grams=200.0,
    )
    assert scaled["serving_grams"] == 200.0
    assert scaled["calories"] == 360.0
    assert scaled["proteins"] == 6.0
    assert scaled["fat"] == 0.6
    assert scaled["carbohydrate"] == 79.6


def test_calculate_nutrition_for_serving_fractional_weight():
    """Verify scaling for 1 butir telur (55g, ratio 0.55)."""
    # 100g telur: 154 kcal, 12.4g protein, 10.8g fat, 0.7g carb
    scaled = calculate_nutrition_for_serving(
        calories_100g=154.0,
        proteins_100g=12.4,
        fat_100g=10.8,
        carb_100g=0.7,
        serving_grams=55.0,
    )
    assert scaled["calories"] == pytest.approx(84.7, abs=0.1)
    assert scaled["proteins"] == pytest.approx(6.8, abs=0.1)
    assert scaled["fat"] == pytest.approx(5.9, abs=0.1)


def test_calculate_nutrition_for_serving_rejects_invalid_inputs():
    """Verify validation errors on non-positive weights or negative nutrients."""
    with pytest.raises(ValueError, match="lebih besar dari 0 gram"):
        calculate_nutrition_for_serving(100.0, 5.0, 2.0, 15.0, serving_grams=0.0)

    with pytest.raises(ValueError, match="tidak boleh negatif"):
        calculate_nutrition_for_serving(-10.0, 5.0, 2.0, 15.0, serving_grams=100.0)


def test_format_serving_summary_outputs_human_readable_text():
    """Verify string formatting output for user display."""
    nutr = {"calories": 180.0, "proteins": 3.0, "fat": 0.3, "carbohydrate": 39.8}
    summary = format_serving_summary("Nasi Putih", "1 Centong (~100g)", nutr)

    assert "Nasi Putih" in summary
    assert "1 Centong (~100g)" in summary
    assert "180.0 kkal" in summary
