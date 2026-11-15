"""Unit tests for BMR, TDEE, biometric validation, and macro calculations."""

from __future__ import annotations

import pytest

from services.nutrition_calc import (
    BiometricValidationError,
    calculate_bmr,
    calculate_macro_split,
    calculate_personalized_nutrition_plan,
    calculate_tdee,
    validate_biometrics,
)


def test_validate_biometrics_happy_path() -> None:
    gender, w, h, a = validate_biometrics("pria", 70.0, 175.0, 25)
    assert gender == "male"
    assert w == 70.0
    assert h == 175.0
    assert a == 25

    gender_f, _, _, _ = validate_biometrics("Wanita", "60", "160", "30")
    assert gender_f == "female"


@pytest.mark.parametrize(
    ("gender", "weight", "height", "age", "expected_msg"),
    [
        ("alien", 70, 170, 25, "Jenis kelamin"),
        ("pria", 10, 170, 25, "Berat badan"),
        ("pria", 70, 30, 25, "Tinggi badan"),
        ("wanita", 60, 160, 5, "Usia"),
        ("pria", "not-a-number", 170, 25, "angka valid"),
    ],
)
def test_validate_biometrics_rejects_out_of_bounds(
    gender: str, weight: float, height: float, age: int, expected_msg: str
) -> None:
    with pytest.raises(BiometricValidationError, match=expected_msg):
        validate_biometrics(gender, weight, height, age)


def test_calculate_bmr_mifflin_st_jeor() -> None:
    # Male: (10 * 70) + (6.25 * 175) - (5 * 25) + 5 = 700 + 1093.75 - 125 + 5 = 1673.75 -> 1673.8
    bmr_m = calculate_bmr("male", 70.0, 175.0, 25)
    assert bmr_m == 1673.8

    # Female: (10 * 60) + (6.25 * 160) - (5 * 30) - 161 = 600 + 1000 - 150 - 161 = 1289.0
    bmr_f = calculate_bmr("female", 60.0, 160.0, 30)
    assert bmr_f == 1289.0


def test_calculate_tdee_scales_with_activity() -> None:
    bmr = 1500.0
    tdee_sedentary = calculate_tdee(bmr, "sedentary")
    assert tdee_sedentary == 1800.0  # 1500 * 1.2

    tdee_moderate = calculate_tdee(bmr, "moderate")
    assert tdee_moderate == 2325.0  # 1500 * 1.55

    with pytest.raises(BiometricValidationError, match="Tingkat aktivitas"):
        calculate_tdee(bmr, "super_extreme")


def test_calculate_macro_split_matches_calorie_ratio() -> None:
    # 2000 kcal:
    # Protein: 25% = 500 kcal / 4 = 125.0g
    # Fat: 25% = 500 kcal / 9 = 55.6g
    # Carb: 50% = 1000 kcal / 4 = 250.0g
    macros = calculate_macro_split(2000.0)
    assert macros["protein_g"] == 125.0
    assert macros["fat_g"] == 55.6
    assert macros["carb_g"] == 250.0


def test_calculate_personalized_nutrition_plan_complete() -> None:
    plan = calculate_personalized_nutrition_plan(
        gender="pria",
        weight_kg=75.0,
        height_cm=175.0,
        age_years=28,
        activity_level="moderate",
        goal="lose_weight",
    )
    assert plan["bmi"] == 24.5
    assert plan["bmi_category"] == "Normal (Ideal)"
    assert plan["bmr"] > 1600.0
    assert plan["tdee"] > plan["bmr"]
    # Deficit should reduce tdee by 500
    assert plan["target_calories"] == round(plan["tdee"] - 500)
    assert "protein_g" in plan["macros"]
