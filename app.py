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
    search_nutrition_keyword_ranked,
)
from services.embeddings import (
    EmbeddingServiceError,
    create_embedding_function,
    get_chroma_client,
    get_collection_count,
    index_nutrition_data,
)
from services.llm import (
    MEDICAL_DISCLAIMER,
    InputValidationError,
    LLMServiceError,
    answer_nutrition_question_stream,
    create_llm_client,
    generate_recipe_stream,
    generate_weekly_menu_stream,
    suggest_healthy_alternatives_stream,
)
from services.nutrition_calc import calculate_personalized_nutrition_plan

logger = get_logger(__name__)


@st.cache_resource
def get_cached_llm_client(api_key: str, base_url: str) -> Any:
    """Cache the OpenAI-compatible LLM client instance across Streamlit reruns."""
    return create_llm_client(api_key=api_key, base_url=base_url)


# Backward-compatible alias
def get_cached_gemini_model(api_key: str, model_name: str) -> Any:
    """Backward-compatible caching function."""
    return create_llm_client(api_key=api_key)


@st.cache_resource
def get_cached_chroma_client(chroma_path: str) -> Any:
    """Cache the ChromaDB PersistentClient across Streamlit reruns."""
    return get_chroma_client(chroma_path)


@st.cache_resource
def get_cached_embedding_fn(api_key: str, base_url: str, model_name: str) -> Any:
    """Cache the OpenAI/SumoPod embedding function across Streamlit reruns."""
    return create_embedding_function(api_key=api_key, base_url=base_url, model_name=model_name)


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
    llm_client: Any,
    chroma_client: Any,
    embedding_fn: Any,
    db_path: str,
    model_name: str,
) -> None:
    """Render the conversational RAG-based nutrition Q&A tab."""
    st.header("💬 Tanya Gizi")
    st.write(
        "Ajukan pertanyaan tentang kandungan gizi makanan berdasarkan dataset nutrisi "
        "(data berbasis standar 100 gram porsi dapat dimakan)."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if st.query_params.get("demo") == "1" and not st.session_state.chat_messages:
        st.session_state.chat_messages = [
            {
                "role": "user",
                "content": "Berapa kalori dan protein dalam tempe goreng vs tahu goreng per 100 gram?",
            },
            {
                "role": "assistant",
                "content": (
                    "Berdasarkan dataset resmi komposisi pangan (standar porsi 100 gram dapat dimakan):\n\n"
                    "1. **Tempe Goreng**:\n"
                    "   - **Energi:** 336 kkal\n"
                    "   - **Protein:** 20.8 gram\n"
                    "   - **Lemak:** 20.6 gram\n"
                    "   - **Karbohidrat:** 17.9 gram\n"
                    "   - **Kalsium:** 155 mg | **Besi:** 4.0 mg\n\n"
                    "2. **Tahu Goreng**:\n"
                    "   - **Energi:** 115 kkal\n"
                    "   - **Protein:** 9.7 gram\n"
                    "   - **Lemak:** 7.3 gram\n"
                    "   - **Karbohidrat:** 2.5 gram\n"
                    "   - **Kalsium:** 168 mg | **Besi:** 2.4 mg\n\n"
                    "💡 **Analisis Nutrisi:**\n"
                    "- Tempe goreng memiliki kepadatan protein lebih dari 2 kali lipat (20.8g vs 9.7g) karena menggunakan biji kedelai utuh yang difermentasi.\n"
                    "- Namun energinya mencapai hampir 3 kali lipat lebih tinggi karena struktur tempe menyerap minyak goreng lebih banyak saat dimasak.\n"
                    "- Jika target diet Anda adalah defisit kalori, tahu goreng atau tempe panggang/kukus merupakan opsi yang lebih efisien kalori.\n\n"
                    f"{MEDICAL_DISCLAIMER}"
                ),
                "contexts": [
                    "Tempe goreng: Energi 336.0 kkal, Protein 20.8 g, Lemak 20.6 g, Karbohidrat 17.9 g, Serat 1.4 g, Kalsium 155.0 mg, Besi 4.0 mg",
                    "Tahu goreng: Energi 115.0 kkal, Protein 9.7 g, Lemak 7.3 g, Karbohidrat 2.5 g, Serat 0.9 g, Kalsium 168.0 mg, Besi 2.4 mg",
                ],
            },
        ]

    # Display past conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("contexts"):
                with st.expander("Referensi Konteks Nutrisi (ChromaDB + SQLite)"):
                    for doc in msg["contexts"]:
                        st.markdown(f"- {doc}")

    # Query Input area
    col_input, col_clear = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Pertanyaan Anda:",
            placeholder="Contoh: Berapa kalori dan protein dalam dada ayam?",
            key="rag_query_input",
        )
    with col_clear:
        if st.button("Hapus Chat", key="clear_chat_button"):
            st.session_state.chat_messages = []
            st.rerun()

    if st.button("Cari Jawaban", key="rag_query_button"):
        cleaned_query = (user_query or "").strip()
        if not cleaned_query:
            st.warning("Pertanyaan tidak boleh kosong.")
            return

        with st.chat_message("user"):
            st.markdown(cleaned_query)

        with st.chat_message("assistant"):
            try:
                stream, contexts = answer_nutrition_question_stream(
                    model=llm_client,
                    query=cleaned_query,
                    chroma_client=chroma_client,
                    embedding_function=embedding_fn,
                    model_name=model_name,
                    db_path=db_path,
                )
                answer_text = st.write_stream(stream)
                st.markdown(MEDICAL_DISCLAIMER)

                if contexts:
                    with st.expander("Referensi Konteks Nutrisi (ChromaDB + SQLite)"):
                        for doc in contexts:
                            st.markdown(f"- {doc}")

                st.session_state.chat_messages.append({"role": "user", "content": cleaned_query})
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": str(answer_text), "contexts": contexts}
                )

            except InputValidationError as exc:
                st.warning(str(exc))
            except EmbeddingServiceError as exc:
                st.warning(str(exc))
                _render_sqlite_fallback(cleaned_query, db_path)
            except LLMServiceError as exc:
                st.error(str(exc))


def _render_recipe_generator(llm_client: Any, model_name: str) -> None:
    """Render the ingredient-based healthy recipe generator tab with diet filters."""
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
    diet_pref = st.selectbox(
        "Preferensi Diet:",
        [
            "Biasa (Tanpa Preferensi)",
            "Tinggi Protein",
            "Rendah Garam (Hipertensi)",
            "Rendah Gula (Diabetes)",
            "Vegetarian",
            "Vegan",
            "Bebas Gluten",
        ],
        key="recipe_diet_pref",
    )

    if st.button("Dapatkan Resep", key="get_recipe_button"):
        with st.spinner("Meracik ide resep sehat..."):
            try:
                stream = generate_recipe_stream(
                    model=llm_client,
                    ingredients=ingredients,
                    diet_preference=diet_pref,
                    model_name=model_name,
                )
                st.divider()
                st.subheader("Ide Resep Anda:")
                recipe_text = st.write_stream(stream)
                st.markdown(MEDICAL_DISCLAIMER)
                st.download_button(
                    "📥 Unduh Resep Ini (Markdown)",
                    data=str(recipe_text),
                    file_name="resep_sehat.md",
                    mime="text/markdown",
                    key="download_recipe_button",
                )
            except InputValidationError as exc:
                st.warning(str(exc))
            except LLMServiceError as exc:
                st.error(str(exc))

    if st.query_params.get("demo") == "1":
        st.divider()
        st.subheader("Ide Resep Anda: Tumis Dada Ayam & Brokoli Saus Gurih")
        st.markdown(
            "**Kategori:** Tinggi Protein & Rendah Karbohidrat  \n"
            "**Porsi:** 2 Porsi | **Waktu Memasak:** 20 Menit\n\n"
            "#### 🥦 Bahan-Bahan:\n"
            "- 200g Dada ayam fillet (potong dadu tipis)\n"
            "- 150g Brokoli segar (potong kuntum, rebus 1 menit)\n"
            "- 1 buah Wortel ukuran sedang (iris serong tipis)\n"
            "- 3 siung Bawang putih & 1/2 bawang bombay (cincang halus)\n"
            "- 1 sdm Saus tiram rendah natrium + 1 sdt minyak zaitun\n\n"
            "#### 🍳 Cara Memasak:\n"
            "1. Panaskan minyak zaitun di wajan, tumis bawang putih dan bombay hingga harum.\n"
            "2. Masukkan potongan dada ayam, masak hingga berubah warna dan matang merata (~5 menit).\n"
            "3. Masukkan wortel dan brokoli, tambahkan saus tiram dan sedikit air hangat.\n"
            "4. Aduk cepat selama 2-3 menit hingga sayuran tetap renyah (*crunchy*). Angkat dan sajikan hangat.\n\n"
            "#### 📊 Estimasi Nilai Gizi Per Porsi:\n"
            "- **Energi:** 265 kkal\n"
            "- **Protein:** 32.5 g\n"
            "- **Lemak:** 6.2 g\n"
            "- **Karbohidrat:** 12.8 g\n\n"
            f"{MEDICAL_DISCLAIMER}"
        )
        st.download_button(
            "📥 Unduh Resep Ini (Markdown)",
            data="Resep Tumis Dada Ayam & Brokoli",
            file_name="resep_sehat.md",
            mime="text/markdown",
            key="demo_recipe_download",
        )


def _render_weekly_planner(llm_client: Any, model_name: str) -> None:
    """Render the 7-day calorie-targeted meal planner tab with BMR/TDEE calculator."""
    st.header("📅 Perencana Menu Mingguan")
    st.write("Buat rencana menu makanan sehat selama 7 hari sesuai kebutuhan kalori harian Anda.")

    expand_calc = st.query_params.get("demo") == "1"
    with st.expander("⚖️ Hitung Kebutuhan Kalori Saya (Kalkulator BMR & TDEE)", expanded=expand_calc):
        c_gender, c_age = st.columns(2)
        with c_gender:
            calc_gender = st.selectbox("Jenis Kelamin:", ["Pria", "Wanita"], key="bmr_gender")
        with c_age:
            calc_age = st.number_input("Usia (tahun):", min_value=10, max_value=120, value=25, key="bmr_age")

        c_w, c_h = st.columns(2)
        with c_w:
            calc_weight = st.number_input(
                "Berat Badan (kg):", min_value=25.0, max_value=250.0, value=65.0, step=0.5, key="bmr_weight"
            )
        with c_h:
            calc_height = st.number_input(
                "Tinggi Badan (cm):", min_value=50.0, max_value=250.0, value=170.0, step=0.5, key="bmr_height"
            )

        calc_activity = st.selectbox(
            "Tingkat Aktivitas Fisik:",
            [
                "Sedentary (Jarang/Tidak Berolahraga)",
                "Light (Olahraga Ringan 1-3 hari/minggu)",
                "Moderate (Olahraga Sedang 3-5 hari/minggu)",
                "Very Active (Olahraga Berat 6-7 hari/minggu)",
            ],
            key="bmr_activity",
        )
        calc_goal = st.selectbox(
            "Tujuan Berat Badan:",
            [
                "Turun Berat Badan (Defisit Sehat -500 kcal)",
                "Pertahankan Berat Badan (Maintenance)",
                "Naik Berat Badan / Bulking (+350 kcal)",
            ],
            key="bmr_goal",
        )

        if st.button("Hitung Kebutuhan Kalori", key="calc_bmr_button") or (
            st.query_params.get("demo") == "1" and "calculated_target_cal" not in st.session_state
        ):
            act_key = "sedentary"
            if "Light" in calc_activity:
                act_key = "light"
            elif "Moderate" in calc_activity:
                act_key = "moderate"
            elif "Very" in calc_activity:
                act_key = "very_active"

            goal_key = "maintain"
            if "Turun" in calc_goal:
                goal_key = "lose_weight"
            elif "Naik" in calc_goal:
                goal_key = "gain_weight"

            plan = calculate_personalized_nutrition_plan(
                gender=calc_gender,
                weight_kg=calc_weight,
                height_cm=calc_height,
                age_years=int(calc_age),
                activity_level=act_key,
                goal=goal_key,
            )

            st.session_state["calculated_target_cal"] = plan["target_calories"]
            st.success(
                f"Hasil: BMR Anda **{plan['bmr']} kcal**, TDEE **{plan['tdee']} kcal**.\n\n"
                f"Rekomendasi Kalori Target: **{plan['target_calories']} kcal/hari** "
                f"(Status BMI: **{plan['bmi_category']}** - {plan['bmi']})."
            )
            col_p, col_f, col_c = st.columns(3)
            col_p.metric("Target Protein", f"{plan['macros']['protein_g']}g")
            col_f.metric("Target Lemak", f"{plan['macros']['fat_g']}g")
            col_c.metric("Target Karbohidrat", f"{plan['macros']['carb_g']}g")

    default_cal = st.session_state.get("calculated_target_cal", DEFAULT_DAILY_CALORIES)
    daily_calories = st.number_input(
        "Target Kalori Harian (kcal):",
        min_value=MIN_DAILY_CALORIES,
        max_value=MAX_DAILY_CALORIES,
        value=int(default_cal),
        step=100,
        key="daily_calories",
    )

    if st.button("Buat Rencana Menu", key="generate_menu_button"):
        with st.spinner("Membuat rencana menu 7 hari..."):
            try:
                stream = generate_weekly_menu_stream(
                    model=llm_client,
                    daily_calories=int(daily_calories),
                    model_name=model_name,
                )
                st.divider()
                st.subheader("Rencana Menu Mingguan Anda:")
                menu_text = st.write_stream(stream)
                st.markdown(MEDICAL_DISCLAIMER)
                st.download_button(
                    "📥 Unduh Menu 7 Hari (Markdown)",
                    data=str(menu_text),
                    file_name="rencana_menu_mingguan.md",
                    mime="text/markdown",
                    key="download_menu_button",
                )
            except InputValidationError as exc:
                st.warning(str(exc))
            except LLMServiceError as exc:
                st.error(str(exc))


def _render_food_comparison(db_path: str) -> None:
    """Render side-by-side nutrition comparison with interactive visual chart."""
    st.header("📊 Bandingkan Nilai Gizi Makanan")
    st.write(
        "Bandingkan kandungan kalori dan makronutrisi antara dua makanan "
        "(data berbasis standar 100 gram porsi dapat dimakan)."
    )

    col1, col2 = st.columns(2)
    with col1:
        food1_query = st.text_input(
            "Makanan Pertama:",
            value="Tempe",
            placeholder="Contoh: Tempe",
            key="compare_food1_input",
        )
    with col2:
        food2_query = st.text_input(
            "Makanan Kedua:",
            value="Tahu",
            placeholder="Contoh: Tahu",
            key="compare_food2_input",
        )

    if st.button("Bandingkan Sekarang", key="compare_button") or (
        st.query_params.get("demo") == "1"
    ):
        matches1 = search_nutrition_keyword_ranked(food1_query, db_path=db_path, limit=1)
        matches2 = search_nutrition_keyword_ranked(food2_query, db_path=db_path, limit=1)

        if not matches1:
            st.warning(f"Makanan pertama '{food1_query}' tidak ditemukan di database.")
            return
        if not matches2:
            st.warning(f"Makanan kedua '{food2_query}' tidak ditemukan di database.")
            return

        f1 = matches1[0]
        f2 = matches2[0]

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f1["name"])
            if f1.get("image") and str(f1["image"]).startswith("http"):
                st.image(f1["image"], width=200)
            st.metric("Energi", f"{f1['calories']} kkal")
            st.write(f"🍗 **Protein:** {f1['proteins']}g")
            st.write(f"🥑 **Lemak:** {f1['fat']}g")
            st.write(f"🍚 **Karbohidrat:** {f1['carbohydrate']}g")

        with c2:
            st.subheader(f2["name"])
            if f2.get("image") and str(f2["image"]).startswith("http"):
                st.image(f2["image"], width=200)
            st.metric("Energi", f"{f2['calories']} kkal")
            st.write(f"🍗 **Protein:** {f2['proteins']}g")
            st.write(f"🥑 **Lemak:** {f2['fat']}g")
            st.write(f"🍚 **Karbohidrat:** {f2['carbohydrate']}g")

        st.subheader("Grafik Perbandingan Makronutrisi (gram)")
        chart_data = pd.DataFrame(
            {
                "Nutrisi": ["Protein (g)", "Lemak (g)", "Karbohidrat (g)"],
                str(f1["name"]): [f1["proteins"], f1["fat"], f1["carbohydrate"]],
                str(f2["name"]): [f2["proteins"], f2["fat"], f2["carbohydrate"]],
            }
        ).set_index("Nutrisi")
        st.bar_chart(chart_data)


def _render_healthy_alternatives(llm_client: Any, model_name: str) -> None:
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
                stream = suggest_healthy_alternatives_stream(
                    model=llm_client,
                    ingredient=ingredient,
                    model_name=model_name,
                )
                st.divider()
                st.subheader(f"Alternatif Sehat untuk '{ingredient.strip()}':")
                st.write_stream(stream)
                st.markdown(MEDICAL_DISCLAIMER)
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
        "Catatan: Proses re-index akan memanggil Embedding API untuk seluruh item "
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
    st.markdown("Asisten pribadi Anda untuk nutrisi, resep sehat, dan kalkulator gizi.")

    try:
        settings = load_settings(require_api_key=True)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        st.error(f"{exc} Salin `.env.example` menjadi `.env` dan isi `API_KEY`.")
        st.stop()

    assert settings.api_key is not None

    try:
        init_sqlite_once(str(settings.csv_path), str(settings.db_path))
    except DatabaseError as exc:
        logger.warning("SQLite auto-initialization skipped: %s", exc)

    try:
        llm_client = get_cached_llm_client(settings.api_key, settings.base_url)
        chroma_client = get_cached_chroma_client(str(settings.chroma_path))
        embedding_fn = get_cached_embedding_fn(
            settings.api_key, settings.base_url, settings.model_embedding
        )
    except Exception:
        logger.exception("Failed initializing application clients.")
        st.error("Gagal menginisialisasi layanan AI atau database vektor. Periksa konfigurasi Anda.")
        st.stop()

    sqlite_count = get_nutrition_count(settings.db_path)
    chroma_count = get_collection_count(chroma_client, embedding_fn)

    menu_options = [
        "Tanya Gizi",
        "Masak Apa Hari Ini?",
        "Perencana Menu Mingguan",
        "Bandingkan Makanan",
        "Alternatif Sehat",
        "Index Data",
    ]
    tab_param = st.query_params.get("tab")
    default_idx = 0
    if tab_param:
        for i, opt in enumerate(menu_options):
            if tab_param.lower() in opt.lower():
                default_idx = i
                break

    with st.sidebar:
        st.header("Menu")
        menu_selection = st.radio(
            "Pilih Opsi",
            menu_options,
            index=default_idx,
        )

        st.divider()
        st.subheader("Informasi Sistem")
        st.write(f"**Model LLM:** `{settings.model_llm}`")
        st.write(f"**Model Embedding:** `{settings.model_embedding}`")
        st.write(f"**Endpoint:** `{settings.base_url}`")
        st.write(f"**Database SQLite:** {sqlite_count} item")
        st.write(f"**Index Vektor:** {chroma_count} dokumen")

        if sqlite_count == 0 or chroma_count == 0:
            st.warning("Data belum diindeks. Buka menu 'Index Data' untuk memuat dataset nutrisi.")

    if menu_selection == "Tanya Gizi":
        _render_tanya_gizi(
            llm_client=llm_client,
            chroma_client=chroma_client,
            embedding_fn=embedding_fn,
            db_path=str(settings.db_path),
            model_name=settings.model_llm,
        )
    elif menu_selection == "Masak Apa Hari Ini?":
        _render_recipe_generator(llm_client=llm_client, model_name=settings.model_llm)
    elif menu_selection == "Perencana Menu Mingguan":
        _render_weekly_planner(llm_client=llm_client, model_name=settings.model_llm)
    elif menu_selection == "Bandingkan Makanan":
        _render_food_comparison(db_path=str(settings.db_path))
    elif menu_selection == "Alternatif Sehat":
        _render_healthy_alternatives(llm_client=llm_client, model_name=settings.model_llm)
    elif menu_selection == "Index Data":
        _render_index_data(
            csv_path=str(settings.csv_path),
            db_path=str(settings.db_path),
            chroma_client=chroma_client,
            embedding_fn=embedding_fn,
        )


if __name__ == "__main__":
    main()
