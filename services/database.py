"""SQLite database management and CSV nutrition dataset ingestion."""

from __future__ import annotations

import re
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
        + " mengandung energi "
        + df["calories"].astype(str)
        + " kkal, protein "
        + df["proteins"].astype(str)
        + "g, lemak "
        + df["fat"].astype(str)
        + "g, dan karbohidrat "
        + df["carbohydrate"].astype(str)
        + "g (per 100g porsi dapat dimakan)."
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


NUTRITION_STOPWORDS: frozenset[str] = frozenset({
    "berapa", "kalori", "dan", "protein", "dalam", "pada", "lemak",
    "karbohidrat", "gizi", "kandungan", "nutrisi", "apa", "apakah",
    "sepiring", "sebutir", "sepotong", "semangkuk", "100g", "100", "gram",
    "makanan", "minuman", "ada", "yang", "di", "ke", "dari", "untuk",
    "dengan", "itu", "ini", "saja", "sih", "dong", "tolong", "bisa",
    "jumlah", "total", "nilai", "kadar", "resep", "sehat", "info",
    "or", "and", "not", "is", "null", "select", "from", "where",
    "like", "order", "by", "limit", "drop", "delete", "insert",
    "update", "table", "true", "false",
})


def extract_food_keywords(query: str) -> list[str]:
    """Extract informative food search keywords from a conversational user query.

    Args:
        query: Raw user query string.

    Returns:
        List of lowercase non-stopword keyword tokens.
    """
    tokens = [w.lower() for w in re.findall(r"\b[A-Za-z0-9-]+\b", query)]
    return [t for t in tokens if t not in NUTRITION_STOPWORDS and len(t) > 1 and not t.isdigit()]



def search_nutrition_keyword_ranked(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search and rank nutrition items by keyword relevance.

    Prioritizes exact full-term matches, followed by multi-token coverage and
    shorter food name proximity.

    Args:
        query: Search query or sentence from user.
        db_path: Path to SQLite database file.
        limit: Maximum number of ranked records to return.

    Returns:
        Ranked list of matching nutrition dictionaries.
    """
    cleaned = query.strip()
    if not cleaned or limit <= 0:
        return []
    if get_nutrition_count(db_path) == 0:
        return []

    tokens = extract_food_keywords(cleaned)
    exact_like = f"%{_escape_like_term(cleaned)}%"

    try:
        with get_db_connection(db_path) as conn:
            # First check for exact whole-phrase match
            exact_rows = conn.execute(
                f"""
                SELECT name, calories, proteins, fat, carbohydrate, description
                FROM {NUTRITION_TABLE_NAME}
                WHERE name LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY LENGTH(name) ASC
                LIMIT ?
                """,
                (exact_like, limit),
            ).fetchall()

            exact_matches = [dict(r) for r in exact_rows]
            if len(exact_matches) >= limit or not tokens:
                return exact_matches

            # Otherwise search across tokens
            conditions = ["name LIKE ? ESCAPE '\\' COLLATE NOCASE" for _ in tokens]
            params = [f"%{_escape_like_term(t)}%" for t in tokens]
            sql = f"""
                SELECT name, calories, proteins, fat, carbohydrate, description
                FROM {NUTRITION_TABLE_NAME}
                WHERE {" OR ".join(conditions)}
            """
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    except sqlite3.Error as exc:
        logger.exception("SQLite ranked keyword search failed for query '%s'", cleaned)
        raise DatabaseError("Gagal mencari data nutrisi berbasis kata kunci.") from exc

    # Score candidates: exact name match = 100, token matches = 10 each, shorter name penalty
    scored: dict[str, tuple[float, dict[str, Any]]] = {}
    for item in exact_matches:
        item_copy = dict(item)
        item_copy["_match_weight"] = 3.5
        scored[item["name"]] = (100.0 - min(len(item["name"]) * 0.1, 10.0), item_copy)

    for item in rows:
        name_lower = str(item["name"]).lower()
        matched_tokens = sum(1 for t in tokens if t in name_lower)
        if matched_tokens == 0:
            continue
        # Base score on token count ratio, penalty for extraneous words
        score = (matched_tokens * 15.0) - min(len(item["name"]) * 0.1, 10.0)
        # Bonus if entire token sequence appears in order
        if all(t in name_lower for t in tokens):
            score += 25.0
        
        weight = 3.0 if matched_tokens >= len(tokens) else (2.0 if matched_tokens > 1 else 1.0)
        item_copy = dict(item)
        item_copy["_match_weight"] = weight

        if item["name"] not in scored or score > scored[item["name"]][0]:
            scored[item["name"]] = (score, item_copy)

    ranked_items = sorted(scored.values(), key=lambda pair: pair[0], reverse=True)
    return [pair[1] for pair in ranked_items[:limit]]


def search_nutrition(
    query: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Backward-compatible nutrition search alias."""
    return search_nutrition_keyword_ranked(query=query, db_path=db_path, limit=limit)



