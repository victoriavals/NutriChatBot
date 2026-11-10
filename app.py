"""Streamlit UI entrypoint for NutriChatBot: Asisten Gizi & Resep Sehat."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config import (
    DEFAULT_DAILY_CALORIES,
    MAX_DAILY_CALORIES,
    MIN_DAILY_CALORIES,
    ConfigurationError,
    get_logger,
    load_settings,
)
from services.database import (
    DatabaseError,
    ensure_sqlite_initialized,
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

logger = get_logger(__name__)


@st.cache_resource
def get_cached_gemini_model(api_key: str, model_name: str) -> Any:
    """Cache the Gemini GenerativeModel instance across Streamlit reruns."""
    return create_gemini_model(api_key=api_key, model_name=model_name)


@st.cache_resource
def get_cached_chroma_client(chroma_path: str) -> Any:
    """Cache the ChromaDB PersistentClient across Streamlit reruns."""
    return get_chroma_client(chroma_path)


@st.cache_resource
def get_cached_embedding_fn(api_key: str) -> Any:
    """Cache the Gemini embedding function across Streamlit reruns."""
    return create_gemini_embedding_function(api_key)


@st.cache_resource
def init_sqlite_once(csv_path: str, db_path: str) -> bool:
    """Auto-populate SQLite from CSV once if the database is missing or empty."""
    return ensure_sqlite_initialized(csv_path=csv_path, db_path=db_path)


def _render_sqlite_fallback(query: str, db_path: str) -> None:
    """Display SQLite substring matches when vector RAG index is unavailable."""
    try:
        matches = search_nutrition(query, db_path=db_path, limit=5)
    except DatabaseError as exc:
        st.error(str(exc))
        return

    if matches:
        st.info("Menampilkan hasil pencarian langsung dari database SQLite:")
        display_df = pd.DataFrame(matches).drop(columns=["description"], errors="ignore")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada kecocokan langsung di database SQLite untuk kata kunci tersebut.")


def _render_tanya_gizi(
    gemini_model: Any,
    chroma_client: Any,
    embedding_fn: Any,
    db_path: str,
) -> None:
    """Render the RAG-based nutrition Q&A tab."""
    st.header("💬 Tanya Gizi")
    st.write("Ajukan pertanyaan tentang kandungan gizi makanan berdasarkan dataset nutrisi.")

    user_query = st.text_input(
        "Pertanyaan Anda:",
        placeholder="Contoh: Berapa kalori sepiring nasi goreng?",
        key="rag_query_input",
    )

    if st.button("Cari Jawaban", key="rag_query_button"):
        with st.spinner("Mencari jawaban..."):
            try:
                answer, contexts = answer_nutrition_question(
                    model=gemini_model,
                    query=user_query,
                    chroma_client=chroma_client,
                    embedding_function=embedding_fn,
                )
                st.divider()
                st.subheader("Jawaban:")
                st.write(answer)
                if contexts:
                    with st.expander("Referensi Konteks Nutrisi (ChromaDB)"):
                        for doc in contexts:
                            st.markdown(f"- {doc}")
            except InputValidationError as exc:
                st.warning(str(exc))
            except EmbeddingServiceError as exc:
                st.warning(str(exc))
                _render_sqlite_fallback(user_query, db_path)
            except LLMServiceError as exc:
                st.error(str(exc))


def _render_recipe_generator(gemini_model: Any) -> None:
    """Render the ingredient-based healthy recipe generator tab."""
    st.header("🍳 Masak Apa Hari Ini?")
    st.write(
        "Berikan daftar bahan-bahan yang Anda miliki (dipisahkan koma), "
        "dan dapatkan ide resep sehat lengkap."
    )

    ingredients = st.text_input(
        "Bahan-bahan:",
        placeholder="Contoh: ayam, brokoli, bawang putih",
        key="recipe_ingredients",
    )

    if st.button("Dapatkan Resep", key="get_recipe_button"):
        with st.spinner("Mencari resep..."):
            try:
                recipe = generate_recipe(model=gemini_model, ingredients=ingredients)
                st.divider()
                st.subheader("Ide Resep Anda:")
                st.write(recipe)
            except InputValidationError as exc:
                st.warning(str(exc))
            except LLMServiceError as exc:
                st.error(str(exc))


def _render_weekly_planner(gemini_model: Any) -> None:
    """Render the 7-day calorie-targeted meal planner tab."""
    st.header("📅 Perencana Menu Mingguan")
    st.write("Buat rencana menu makanan sehat selama 7 hari sesuai kebutuhan kalori harian Anda.")

    daily_calories = st.number_input(
        "Target Kalori Harian (kcal):",
        min_value=MIN_DAILY_CALORIES,
        max_value=MAX_DAILY_CALORIES,
        value=DEFAULT_DAILY_CALORIES,
        step=100,
        key="daily_calories",
    )

    if st.button("Buat Rencana Menu", key="generate_menu_button"):
        with st.spinner("Membuat rencana menu..."):
            try:
                menu_plan = generate_weekly_menu(
                    model=gemini_model,
                    daily_calories=int(daily_calories),
                )
                st.divider()
                st.subheader("Rencana Menu Mingguan Anda:")
                st.write(menu_plan)
            except InputValidationError as exc:
                st.warning(str(exc))
            except LLMServiceError as exc:
                st.error(str(exc))


def _render_healthy_alternatives(gemini_model: Any) -> None:
    """Render the healthy ingredient substitution tab."""
    st.header("🥗 Alternatif Sehat")
    st.write("Masukkan bahan makanan yang ingin Anda cari alternatif lebih sehatnya.")

    ingredient = st.text_input(
        "Bahan yang ingin diganti:",
        placeholder="Contoh: gula, mentega, tepung terigu",
        key="healthy_alternative_input",
    )

    if st.button("Cari Alternatif", key="find_alternative_button"):
        with st.spinner("Mencari alternatif..."):
            try:
                alternatives = suggest_healthy_alternatives(
                    model=gemini_model,
                    ingredient=ingredient,
                )
                st.divider()
                st.subheader(f"Alternatif Sehat untuk '{ingredient.strip()}':")
                st.write(alternatives)
            except InputValidationError as exc:
                st.warning(str(exc))
            except LLMServiceError as exc:
                st.error(str(exc))


def _render_index_data(
    csv_path: str,
    db_path: str,
    chroma_client: Any,
    embedding_fn: Any,
) -> None:
    """Render the dataset ingestion and ChromaDB re-indexing administration tab."""
    st.header("⚙️ Index Data Nutrisi")
    st.write(
        "Muat ulang dataset dari `data/nutrition.csv` ke database SQLite dan bangun ulang "
        "index vektor ChromaDB untuk fitur Tanya Gizi (RAG)."
    )
    st.info(
        "Catatan: Proses re-index akan memanggil Gemini Embedding API untuk seluruh item "
        "di dataset CSV."
    )

    if st.button("Muat & Indeks Ulang Data", key="reindex_button"):
        with st.spinner("Memuat CSV, menyimpan ke SQLite, dan mengindeks ke ChromaDB..."):
            try:
                df, anomalies = load_and_validate_csv(csv_path)
                sqlite_count = save_nutrition_to_sqlite(df, db_path)
                chroma_count = index_nutrition_data(
                    df=df,
                    client=chroma_client,
                    embedding_function=embedding_fn,
                )
                st.success(
                    f"Berhasil memuat {sqlite_count} item ke SQLite dan mengindeks "
                    f"{chroma_count} item ke ChromaDB."
                )
                if anomalies:
                    with st.expander(f"Peringatan Anomali Data Sumber ({len(anomalies)} item)"):
                        for warning_msg in anomalies:
                            st.write(f"- {warning_msg}")
            except (DatabaseError, EmbeddingServiceError, ValueError) as exc:
                st.error(str(exc))


def main() -> None:
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="NutriChatBot: Asisten Gizi & Resep Sehat",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("NutriChatBot: Asisten Gizi & Resep Sehat 🍎🥕")
    st.markdown("Asisten pribadi Anda untuk nutrisi dan resep sehat.")

    try:
        settings = load_settings(require_api_key=True)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        st.error(f"{exc} Salin `.env.example` menjadi `.env` dan isi `GEMINI_API_KEY`.")
        st.stop()

    assert settings.gemini_api_key is not None

    try:
        init_sqlite_once(str(settings.csv_path), str(settings.db_path))
    except DatabaseError as exc:
        logger.warning("SQLite auto-initialization skipped: %s", exc)

    try:
        gemini_model = get_cached_gemini_model(settings.gemini_api_key, settings.gemini_model)
        chroma_client = get_cached_chroma_client(str(settings.chroma_path))
        embedding_fn = get_cached_embedding_fn(settings.gemini_api_key)
    except Exception:
        logger.exception("Failed initializing application clients.")
        st.error("Gagal menginisialisasi layanan AI atau database vektor. Periksa konfigurasi Anda.")
        st.stop()

    sqlite_count = get_nutrition_count(settings.db_path)
    chroma_count = get_collection_count(chroma_client, embedding_fn)

    with st.sidebar:
        st.header("Menu")
        menu_selection = st.radio(
            "Pilih Opsi",
            [
                "Tanya Gizi",
                "Masak Apa Hari Ini?",
                "Perencana Menu Mingguan",
                "Alternatif Sehat",
                "Index Data",
            ],
        )
        st.divider()
        st.caption(f"SQLite: {sqlite_count} item | ChromaDB RAG: {chroma_count} item")
        if chroma_count == 0:
            st.warning(
                "Index ChromaDB masih kosong (0 item). Gunakan menu **Index Data** atau "
                "`scripts/rebuild_index.py` untuk mengaktifkan RAG."
            )
        st.divider()
        st.write("By Naufal Firdaus")

    if menu_selection == "Tanya Gizi":
        _render_tanya_gizi(gemini_model, chroma_client, embedding_fn, str(settings.db_path))
    elif menu_selection == "Masak Apa Hari Ini?":
        _render_recipe_generator(gemini_model)
    elif menu_selection == "Perencana Menu Mingguan":
        _render_weekly_planner(gemini_model)
    elif menu_selection == "Alternatif Sehat":
        _render_healthy_alternatives(gemini_model)
    elif menu_selection == "Index Data":
        _render_index_data(
            str(settings.csv_path),
            str(settings.db_path),
            chroma_client,
            embedding_fn,
        )


if __name__ == "__main__":
    main()
