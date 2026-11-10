"""Integration checks for the repository's nutrition dataset and rebuild CLI."""

from __future__ import annotations

from pathlib import Path

from config import DEFAULT_CSV_PATH
from scripts.rebuild_index import clean_chroma_directory
from scripts.rebuild_index import main as rebuild_main
from services.database import get_nutrition_count, load_and_validate_csv


def test_repository_nutrition_csv_integrity() -> None:
    df, anomalies = load_and_validate_csv(DEFAULT_CSV_PATH)
    assert len(df) == 1346
    assert {"id", "calories", "proteins", "fat", "carbohydrate", "name", "description"}.issubset(
        df.columns
    )
    assert df["name"].isna().sum() == 0
    # Source dataset contains known anomalies; ensure detection runs deterministically
    assert isinstance(anomalies, list)
    assert len(anomalies) >= 1


def test_rebuild_index_cli_skip_chroma_builds_sqlite(
    sample_csv_file: Path,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "cli_nutrition.db"
    chroma_dir = tmp_path / "chroma_dir"
    stale_dir = chroma_dir / "orphaned-segment"
    stale_dir.mkdir(parents=True)
    (stale_dir / "junk.bin").write_bytes(b"123")

    exit_code = rebuild_main(
        [
            "--csv",
            str(sample_csv_file),
            "--db",
            str(db_path),
            "--chroma",
            str(chroma_dir),
            "--skip-chroma",
        ]
    )
    assert exit_code == 0
    assert get_nutrition_count(db_path) == 3

    clean_chroma_directory(chroma_dir)
    assert list(chroma_dir.iterdir()) == []

