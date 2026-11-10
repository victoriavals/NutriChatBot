"""CLI utility to rebuild SQLite nutrition database and ChromaDB vector index."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import DEFAULT_CHROMA_PATH, DEFAULT_CSV_PATH, DEFAULT_DB_PATH, get_logger, load_settings
from services.database import load_and_validate_csv, save_nutrition_to_sqlite
from services.embeddings import (
    create_gemini_embedding_function,
    get_chroma_client,
    index_nutrition_data,
)

logger = get_logger(__name__)


def clean_chroma_directory(chroma_path: Path) -> None:
    """Remove stale or orphaned files inside the ChromaDB directory safely."""
    resolved = chroma_path.resolve()
    if not resolved.exists():
        return
    if resolved == Path("/") or resolved == resolved.parent:
        raise ValueError(f"Refusing to clean unsafe root path: {resolved}")

    for child in resolved.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    logger.info("Cleaned existing contents of ChromaDB directory at %s.", resolved)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for index rebuilding."""
    parser = argparse.ArgumentParser(
        description="Rebuild NutriChatBot SQLite database and ChromaDB vector index."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to nutrition CSV dataset.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database path.",
    )
    parser.add_argument(
        "--chroma",
        type=Path,
        default=DEFAULT_CHROMA_PATH,
        help="Target ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing ChromaDB segments/files before indexing.",
    )
    parser.add_argument(
        "--skip-chroma",
        action="store_true",
        help="Only rebuild SQLite database without calling Gemini Embeddings API.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute SQLite and optional ChromaDB indexing from CSV."""
    args = parse_args(argv)
    settings = load_settings(require_api_key=not args.skip_chroma)

    df, anomalies = load_and_validate_csv(args.csv)
    sqlite_rows = save_nutrition_to_sqlite(df, args.db)
    logger.info("SQLite rebuild complete: %d rows written.", sqlite_rows)

    if anomalies:
        logger.warning("Found %d data anomaly warnings in %s.", len(anomalies), args.csv)

    if args.skip_chroma:
        logger.info("Skipping ChromaDB embedding indexing (--skip-chroma).")
        return 0

    if args.clean:
        clean_chroma_directory(args.chroma)

    assert settings.gemini_api_key is not None
    embedding_fn = create_gemini_embedding_function(settings.gemini_api_key)
    client = get_chroma_client(args.chroma)
    indexed = index_nutrition_data(df=df, client=client, embedding_function=embedding_fn)
    logger.info("ChromaDB rebuild complete: %d documents indexed.", indexed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

