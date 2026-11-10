"""ChromaDB vector indexing and RAG retrieval service."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    CHROMA_COLLECTION_NAME,
    DEFAULT_CHROMA_PATH,
    DEFAULT_RAG_RESULTS,
    EMBEDDING_BATCH_SIZE,
    get_logger,
)

logger = get_logger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when vector indexing or semantic retrieval fails."""


def ensure_pysqlite3() -> bool:
    """Swap stdlib sqlite3 with pysqlite3 on Linux systems with older SQLite builds.

    Returns:
        True if pysqlite3 was activated, False otherwise.
    """
    if sys.platform != "linux":
        return False
    try:
        pysqlite3_mod = __import__("pysqlite3")
        sys.modules["sqlite3"] = pysqlite3_mod
        return True
    except ImportError:
        logger.debug("pysqlite3 not installed; using built-in sqlite3.")
        return False


def create_gemini_embedding_function(api_key: str) -> Any:
    """Lazily create a ChromaDB Google Generative AI embedding function.

    Args:
        api_key: Non-empty Gemini API key.

    Returns:
        Configured GoogleGenerativeAiEmbeddingFunction instance.

    Raises:
        ValueError: If api_key is empty.
    """
    if not api_key or not api_key.strip():
        raise ValueError("GEMINI_API_KEY tidak boleh kosong untuk inisialisasi embedding.")

    from chromadb.utils import embedding_functions

    return embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key.strip())


def get_chroma_client(chroma_path: str | Path = DEFAULT_CHROMA_PATH) -> Any:
    """Initialize and return a persistent ChromaDB client (SQLite + HNSW).

    Args:
        chroma_path: Directory where ChromaDB persists vector segments and metadata.

    Returns:
        chromadb.PersistentClient instance.
    """
    ensure_pysqlite3()
    import chromadb

    path = Path(chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def _sanitize_metadata_value(value: Any) -> str | int | float | bool:
    """Convert DataFrame metadata cell values into ChromaDB-safe primitives."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return 0.0 if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, int):
        return value
    if hasattr(value, "item"):
        return _sanitize_metadata_value(value.item())
    if pd.isna(value):
        return ""
    return str(value)


def sanitize_metadata_records(df: pd.DataFrame) -> list[dict[str, str | int | float | bool]]:
    """Convert non-description columns into ChromaDB-compatible metadata dictionaries."""
    meta_df = df.drop(columns=["description"], errors="ignore")
    raw_records = meta_df.to_dict(orient="records")
    return [
        {str(key): _sanitize_metadata_value(val) for key, val in record.items()}
        for record in raw_records
    ]


def _delete_collection_if_exists(client: Any, collection_name: str) -> None:
    """Delete a ChromaDB collection if it exists without swallowing unexpected errors."""
    try:
        existing_names = {
            getattr(col, "name", col) for col in client.list_collections()
        }
        if collection_name in existing_names:
            client.delete_collection(name=collection_name)
            logger.info("Deleted existing ChromaDB collection '%s'.", collection_name)
    except ValueError as exc:
        logger.debug("Collection '%s' did not exist before delete: %s", collection_name, exc)


def index_nutrition_data(
    df: pd.DataFrame,
    client: Any,
    embedding_function: Any,
    collection_name: str = CHROMA_COLLECTION_NAME,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> int:
    """Index nutrition descriptions into ChromaDB in controlled batches.

    Args:
        df: DataFrame containing a 'description' column and metadata columns.
        client: Active ChromaDB client.
        embedding_function: Embedding callable/function for ChromaDB collection.
        collection_name: Target ChromaDB collection name.
        batch_size: Number of documents per batch insert.

    Returns:
        Total number of indexed documents.

    Raises:
        ValueError: If 'description' column is missing or batch_size <= 0.
        EmbeddingServiceError: If indexing into ChromaDB fails.
    """
    if "description" not in df.columns:
        raise ValueError("DataFrame wajib memiliki kolom 'description' untuk indexing RAG.")
    if batch_size <= 0:
        raise ValueError("batch_size harus lebih besar dari 0.")

    documents = [str(doc) for doc in df["description"].tolist()]
    if not documents:
        logger.warning("No nutrition documents to index.")
        return 0

    metadatas = sanitize_metadata_records(df)
    ids = [f"doc_{idx}" for idx in range(len(documents))]

    try:
        _delete_collection_if_exists(client, collection_name)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
            logger.info("Indexed batch %d-%d of %d nutrition items.", start + 1, end, total)
        return total
    except Exception as exc:
        logger.exception("Failed indexing nutrition data into ChromaDB.")
        raise EmbeddingServiceError("Gagal mengindeks data nutrisi ke ChromaDB.") from exc


def _chroma_recoverable_errors() -> tuple[type[BaseException], ...]:
    """Return ChromaDB and IO exception types expected on missing/unreadable stores."""
    try:
        from chromadb.errors import ChromaError

        return (ChromaError, ValueError, OSError)
    except ImportError:
        return (ValueError, OSError)


def get_collection_count(
    client: Any,
    embedding_function: Any | None = None,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> int:
    """Return number of documents in the ChromaDB collection, or 0 if missing/unreadable."""
    recoverable = _chroma_recoverable_errors()
    try:
        existing_names = {getattr(col, "name", col) for col in client.list_collections()}
        if collection_name not in existing_names:
            return 0
        kwargs: dict[str, Any] = {"name": collection_name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        collection = client.get_collection(**kwargs)
        return int(collection.count())
    except recoverable as exc:
        logger.debug("ChromaDB collection '%s' unavailable: %s", collection_name, exc)
        return 0


def retrieve_nutrition_documents(
    query: str,
    client: Any,
    embedding_function: Any,
    collection_name: str = CHROMA_COLLECTION_NAME,
    n_results: int = DEFAULT_RAG_RESULTS,
) -> list[str]:
    """Retrieve top-N semantically similar nutrition descriptions from ChromaDB.

    Args:
        query: User's nutrition question.
        client: Active ChromaDB client.
        embedding_function: Configured embedding function matching the collection.
        collection_name: Target ChromaDB collection name.
        n_results: Maximum number of contexts to retrieve.

    Returns:
        List of matching nutrition description strings.

    Raises:
        EmbeddingServiceError: If collection is missing, empty, or query execution fails.
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )
        doc_count = int(collection.count())
    except Exception as exc:
        logger.warning("Failed accessing ChromaDB collection '%s': %s", collection_name, exc)
        raise EmbeddingServiceError(
            "Database vektor nutrisi belum tersedia. Silakan jalankan proses Index Data terlebih dahulu."
        ) from exc

    if doc_count == 0:
        raise EmbeddingServiceError(
            "Database vektor nutrisi masih kosong (0 dokumen). Silakan jalankan Index Data terlebih dahulu."
        )

    effective_results = max(1, min(n_results, doc_count))
    try:
        results = collection.query(
            query_texts=[cleaned_query],
            n_results=effective_results,
        )
    except Exception as exc:
        logger.exception("ChromaDB query failed.")
        raise EmbeddingServiceError("Gagal mengambil konteks nutrisi dari database vektor.") from exc

    documents: list[str] = []
    raw_docs = results.get("documents") if isinstance(results, dict) else None
    if raw_docs:
        for doc_group in raw_docs:
            for doc in doc_group:
                if isinstance(doc, str) and doc.strip():
                    documents.append(doc)
    return documents

