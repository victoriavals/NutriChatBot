"""Service layer package for NutriChatBot."""

from services.database import (
    DatabaseError,
    detect_nutrition_anomalies,
    ensure_sqlite_initialized,
    get_db_connection,
    get_nutrition_count,
    load_and_validate_csv,
    save_nutrition_to_sqlite,
    search_nutrition,
)
from services.embeddings import (
    EmbeddingServiceError,
    create_gemini_embedding_function,
    get_chroma_client,
    get_collection_count,
    index_nutrition_data,
    retrieve_nutrition_documents,
)
from services.llm import (
    InputValidationError,
    LLMServiceError,
    answer_nutrition_question,
    create_gemini_model,
    generate_recipe,
    generate_weekly_menu,
    suggest_healthy_alternatives,
)

__all__ = [
    "DatabaseError",
    "EmbeddingServiceError",
    "InputValidationError",
    "LLMServiceError",
    "answer_nutrition_question",
    "create_gemini_embedding_function",
    "create_gemini_model",
    "detect_nutrition_anomalies",
    "ensure_sqlite_initialized",
    "generate_recipe",
    "generate_weekly_menu",
    "get_chroma_client",
    "get_collection_count",
    "get_db_connection",
    "get_nutrition_count",
    "index_nutrition_data",
    "load_and_validate_csv",
    "retrieve_nutrition_documents",
    "save_nutrition_to_sqlite",
    "search_nutrition",
    "suggest_healthy_alternatives",
]

