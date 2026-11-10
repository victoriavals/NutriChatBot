"""SQLite database management and CSV nutrition dataset ingestion."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from config import DEFAULT_CSV_PATH, DEFAULT_DB_PATH, get_logger

logger = get_logger(__name__)

REQUIRED_NUTRITION_COLUMNS: tuple[str, ...] = (
    "name",
    "calories",
    "proteins",
    "fat",
    "carbohydrate",
)
NUMERIC_NUTRITION_COLUMNS: tuple[str, ...] = (
    "calories",
    "proteins",
    "fat",
    "carbohydrate",
)
NUTRITION_TABLE_NAME: str = "nutrition"


class DatabaseError(Exception):
    """Raised when SQLite operations or CSV ingestion fail."""


@contextmanager
def get_db_connection(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Context manager that yields an SQLite connection and guarantees closure.

    Args:
        db_path: Path to the SQLite database file.

    Yields:
        Active sqlite3.Connection configured with sqlite3.Row row_factory.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def detect_nutrition_anomalies(df: pd.DataFrame) -> list[str]:
    """Inspect a normalized nutrition DataFrame for implausible nutrient values.

    Args:
        df: DataFrame containing normalized nutrition columns.

    Returns:
        List of human-readable anomaly warning strings.
    """
    warnings: list[str] = []
    if df.empty:
        return warnings

    for _, row in df.iterrows():
        name = str(row.get("name", "Unknown"))
        calories = float(row.get("calories", 0.0))
        proteins = float(row.get("proteins", 0.0))
        fat = float(row.get("fat", 0.0))
        carb = float(row.get("carbohydrate", 0.0))

        if any(val < 0 for val in (calories, proteins, fat, carb)):
            warnings.append(f"Negative nutrition value on '{name}'.")
        elif calories == 0 and (proteins > 0 or fat > 0 or carb > 0):
            warnings.append(f"Zero calories with non-zero macronutrients on '{name}'.")
        elif calories > 0 and (carb * 4.0) > (calories * 1.5):
            warnings.append(
                f"Implausible carb-to-calorie ratio on '{name}': "
                f"{calories} kcal vs {carb}g carbohydrate."
            )

    return warnings


def load_and_validate_csv(csv_path: str | Path = DEFAULT_CSV_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Load nutrition CSV, normalize schema, build RAG descriptions, and check anomalies.

    Args:
        csv_path: Path to the nutrition CSV dataset.

    Returns:
        Tuple of (normalized DataFrame with 'description', anomaly warning list).

    Raises:
        DatabaseError: If CSV is missing, empty, unreadable, or lacks required columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise DatabaseError(f"Dataset CSV tidak ditemukan di '{path}'.")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.exception("Failed to read CSV file at %s", path)
        raise DatabaseError(f"Gagal membaca file CSV: {path.name}") from exc

    if df.empty:
        raise DatabaseError("Dataset CSV kosong.")

    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")

    missing_cols = [col for col in REQUIRED_NUTRITION_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DatabaseError(
            f"Kolom wajib tidak ditemukan di CSV: {', '.join(missing_cols)}"
        )

    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df = df[df["name"] != ""].copy()
    if df.empty:
        raise DatabaseError("Dataset CSV tidak memiliki baris makanan yang valid.")

    for col in NUMERIC_NUTRITION_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        invalid_count = int(coerced.isna().sum())
        if invalid_count > 0:
            logger.warning("Coerced %d invalid values in column '%s' to 0.0.", invalid_count, col)
        df[col] = coerced.fillna(0.0)

    if "image" in df.columns:
        df["image"] = df["image"].fillna("").astype(str)

    df["description"] = (
        df["name"]
        + " has "
        + df["calories"].astype(str)
        + " calories, "
        + df["proteins"].astype(str)
        + "g protein, "
        + df["fat"].astype(str)
        + "g fat, and "
        + df["carbohydrate"].astype(str)
        + "g carbohydrate."
    )

    anomalies = detect_nutrition_anomalies(df)
    if anomalies:
        logger.warning("Detected %d nutritional data anomalies in %s.", len(anomalies), path.name)

    return df, anomalies


def save_nutrition_to_sqlite(
    df: pd.DataFrame,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Persist validated nutrition DataFrame into SQLite and build lookup index.

    Args:
        df: Validated DataFrame produced by load_and_validate_csv.
        db_path: Target SQLite database path.

    Returns:
        Number of rows written to SQLite.
    """
    try:
        with get_db_connection(db_path) as conn:
            df.to_sql(NUTRITION_TABLE_NAME, conn, if_exists="replace", index=False)
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_nutrition_name "
                f"ON {NUTRITION_TABLE_NAME}(name COLLATE NOCASE)"
            )
    except sqlite3.Error as exc:
        logger.exception("Failed saving nutrition data to SQLite at %s", db_path)
        raise DatabaseError("Gagal menyimpan data nutrisi ke SQLite.") from exc

    row_count = len(df)
    logger.info("Stored %d nutrition items into SQLite database at %s.", row_count, db_path)
    return row_count


def get_nutrition_count(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Return total rows stored in the SQLite nutrition table, or 0 if unavailable."""
    path = Path(db_path)
    if not path.exists():
        return 0
    try:
        with get_db_connection(path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (NUTRITION_TABLE_NAME,),
            )
            if cursor.fetchone() is None:
                return 0
            count_row = conn.execute(f"SELECT COUNT(*) FROM {NUTRITION_TABLE_NAME}").fetchone()
            return int(count_row[0]) if count_row else 0
    except sqlite3.Error as exc:
        logger.warning("Could not read nutrition count from SQLite: %s", exc)
        return 0


def ensure_sqlite_initialized(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> bool:
    """Initialize SQLite from CSV automatically if the database/table is missing or empty.

    Returns:
        True if initialization was performed, False if data already existed.
    """
    if get_nutrition_count(db_path) > 0:
        return False
    df, _ = load_and_validate_csv(csv_path)
    save_nutrition_to_sqlite(df, db_path)
    return True


def _escape_like_term(term: str) -> str:
    """Escape SQL LIKE wildcard characters so user input is treated literally."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_nutrition(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search nutrition items in SQLite by food name (case-insensitive substring match).

    Args:
        query: Search keyword provided by the user.
        db_path: Path to SQLite database file.
        limit: Maximum number of records to return.

    Returns:
        List of matching nutrition dictionaries.
    """
    cleaned = query.strip()
    if not cleaned or limit <= 0:
        return []
    if get_nutrition_count(db_path) == 0:
        return []

    pattern = f"%{_escape_like_term(cleaned)}%"
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT name, calories, proteins, fat, carbohydrate, description
                FROM {NUTRITION_TABLE_NAME}
                WHERE name LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY name ASC
                LIMIT ?
                """,
                (pattern, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        logger.exception("SQLite search failed for query '%s'", cleaned)
        raise DatabaseError("Gagal mencari data nutrisi di database.") from exc

