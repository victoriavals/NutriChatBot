"""Personalized nutrition calculations: BMR (Mifflin-St Jeor), TDEE, and macro distribution."""

from __future__ import annotations

from typing import Any

from config import get_logger

logger = get_logger(__name__)

# Valid activity levels and their corresponding physical activity factors (PAL)
ACTIVITY_FACTORS: dict[str, float] = {
    "sedentary": 1.2,  # Minimal/no exercise, desk job
    "light": 1.375,  # Light exercise 1-3 days/week
    "moderate": 1.55,  # Moderate exercise 3-5 days/week
    "very_active": 1.725,  # Hard exercise 6-7 days/week
    "extra_active": 1.9,  # Very hard daily exercise or physical job
}

# Standard calorie adjustment based on body weight goal
GOAL_CALORIE_ADJUSTMENTS: dict[str, int] = {
    "lose_weight": -500,  # Sustainable fat loss deficit (~0.5 kg/week)
    "maintain": 0,  # Weight maintenance
    "gain_weight": 350,  # Lean surplus for muscle gain
}


class BiometricValidationError(ValueError):
    """Raised when biometric inputs fall outside plausible physiological limits."""


def validate_biometrics(
    gender: str,
    weight_kg: float,
    height_cm: float,
    age_years: int,
) -> tuple[str, float, float, int]:
    """Validate physiological plausibility of user inputs.

    Args:
        gender: "male" / "pria" or "female" / "wanita".
        weight_kg: Weight in kilograms (25.0 - 300.0 kg).
        height_cm: Height in centimeters (50.0 - 250.0 cm).
        age_years: Age in years (10 - 120 years).

    Returns:
        Normalized tuple (gender_normalized, weight_kg, height_cm, age_years).

    Raises:
        BiometricValidationError: If any value is out of physiological range.
    """
    cleaned_gender = str(gender).strip().lower()
    if cleaned_gender in ("pria", "male", "l", "laki-laki", "m"):
        norm_gender = "male"
    elif cleaned_gender in ("wanita", "female", "p", "perempuan", "f"):
        norm_gender = "female"
    else:
        raise BiometricValidationError(
            "Jenis kelamin harus 'pria' atau 'wanita'."
        )

    try:
        w = float(weight_kg)
        h = float(height_cm)
        a = int(age_years)
    except (TypeError, ValueError) as exc:
        raise BiometricValidationError("Berat, tinggi, dan usia harus berupa angka valid.") from exc

    if w < 25.0 or w > 300.0:
        raise BiometricValidationError("Berat badan harus antara 25 kg dan 300 kg.")
    if h < 50.0 or h > 250.0:
        raise BiometricValidationError("Tinggi badan harus antara 50 cm dan 250 cm.")
    if a < 10 or a > 120:
        raise BiometricValidationError("Usia harus antara 10 tahun dan 120 tahun.")

    return norm_gender, w, h, a


def calculate_bmr(
    gender: str,
    weight_kg: float,
    height_cm: float,
    age_years: int,
) -> float:
    """Calculate Basal Metabolic Rate using the Mifflin-St Jeor equation.

    Formula:
        Male: 10 * weight (kg) + 6.25 * height (cm) - 5 * age (y) + 5
        Female: 10 * weight (kg) + 6.25 * height (cm) - 5 * age (y) - 161

    Returns:
        Calculated BMR in kilocalories per day (rounded to 1 decimal place).
    """
    norm_gender, w, h, a = validate_biometrics(gender, weight_kg, height_cm, age_years)

    base = (10.0 * w) + (6.25 * h) - (5.0 * a)
    bmr = base + 5.0 if norm_gender == "male" else base - 161.0
    return round(bmr, 1)


def calculate_tdee(
    bmr: float,
    activity_level: str = "sedentary",
) -> float:
    """Calculate Total Daily Energy Expenditure by scaling BMR with activity factor.

    Args:
        bmr: Basal Metabolic Rate in kcal.
        activity_level: Key in ACTIVITY_FACTORS (e.g. sedentary, light, moderate).

    Returns:
        Calculated TDEE in kcal (rounded to 1 decimal place).
    """
    if bmr <= 0:
        raise BiometricValidationError("BMR harus bernilai positif.")

    norm_activity = str(activity_level).strip().lower()
    factor = ACTIVITY_FACTORS.get(norm_activity)
    if factor is None:
        valid_keys = ", ".join(ACTIVITY_FACTORS.keys())
        raise BiometricValidationError(
            f"Tingkat aktivitas tidak valid '{activity_level}'. Pilihan: {valid_keys}"
        )

    return round(bmr * factor, 1)


def calculate_macro_split(daily_calories: float) -> dict[str, float]:
    """Calculate balanced macronutrient targets in grams based on standard dietary guidelines.

    Macro ratio:
        Protein: 25% of calories (4 kcal/g)
        Fat: 25% of calories (9 kcal/g)
        Carbohydrates: 50% of calories (4 kcal/g)

    Returns:
        Dictionary with grams for protein_g, fat_g, and carb_g.
    """
    cal = max(500.0, float(daily_calories))
    protein_grams = round((cal * 0.25) / 4.0, 1)
    fat_grams = round((cal * 0.25) / 9.0, 1)
    carb_grams = round((cal * 0.50) / 4.0, 1)

    return {
        "protein_g": protein_grams,
        "fat_g": fat_grams,
        "carb_g": carb_grams,
    }


def calculate_personalized_nutrition_plan(
    gender: str,
    weight_kg: float,
    height_cm: float,
    age_years: int,
    activity_level: str = "sedentary",
    goal: str = "maintain",
) -> dict[str, Any]:
    """Generate a complete daily energy and macronutrient blueprint.

    Args:
        gender: "male" / "female"
        weight_kg: User weight in kg
        height_cm: User height in cm
        age_years: User age in years
        activity_level: "sedentary", "light", "moderate", "very_active", "extra_active"
        goal: "lose_weight", "maintain", "gain_weight"

    Returns:
        Dictionary containing BMR, TDEE, target_calories, macro splits, and human-readable explanation.
    """
    bmr = calculate_bmr(gender, weight_kg, height_cm, age_years)
    tdee = calculate_tdee(bmr, activity_level)

    norm_goal = str(goal).strip().lower()
    adjustment = GOAL_CALORIE_ADJUSTMENTS.get(norm_goal, 0)
    target_calories = max(1000, round(tdee + adjustment))

    macros = calculate_macro_split(float(target_calories))

    bmi = round(weight_kg / ((height_cm / 100.0) ** 2), 1)

    if bmi < 18.5:
        bmi_category = "Kurus (Underweight)"
    elif bmi < 25.0:
        bmi_category = "Normal (Ideal)"
    elif bmi < 30.0:
        bmi_category = "Kelebihan Berat Badan (Overweight)"
    else:
        bmi_category = "Obesitas"

    return {
        "bmi": bmi,
        "bmi_category": bmi_category,
        "bmr": bmr,
        "tdee": tdee,
        "goal": norm_goal,
        "target_calories": target_calories,
        "macros": macros,
    }
