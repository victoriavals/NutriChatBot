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
    get_packaged_categories,
    get_packaged_nutrition_count,
    get_recipes_count,
    load_and_validate_csv,
    save_nutrition_to_sqlite,
    search_nutrition,
    search_nutrition_keyword_ranked,
    search_packaged_foods,
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
from services.recipe_service import find_recipes_by_pantry_ingredients
from services.serving_size import (
    calculate_nutrition_for_serving,
    format_serving_summary,
    get_serving_options,
)

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


def _inject_custom_styles() -> None:
    """Inject clean, modern, user-friendly CSS styles into Streamlit app."""
    st.markdown(
        """
        <style>
        /* Modern container spacing */
        .main .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.5rem;
            max-width: 1100px;
        }
        /* Metric card styling */
        div[data-testid="stMetric"] {
            background-color: rgba(28, 38, 56, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 12px;
            padding: 14px 18px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        div[data-testid="stMetric"]:hover {
            border-color: rgba(59, 130, 246, 0.5);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.88rem;
            font-weight: 600;
            color: #94a3b8;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
            font-weight: 700;
        }
        /* Friendly button styling */
        .stButton button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }
        /* Clean chat message borders */
        .stChatMessage {
            border-radius: 12px;
            padding: 12px;
            margin-bottom: 8px;
        }
        /* Nutri-Score Badges */
        .nutri-score-pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 0.95rem;
            color: #ffffff;
            letter-spacing: 0.5px;
        }
        .nutri-a { background-color: #038141; }
        .nutri-b { background-color: #85BB2F; }
        .nutri-c { background-color: #FECB02; color: #222222; }
        .nutri-d { background-color: #EE8100; }
        .nutri-e { background-color: #E63E11; }
        .fmcg-card {
            background-color: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    # Handle quick suggestion click if selected
    if st.session_state.get("suggested_rag_query"):
        st.session_state["rag_query_input"] = st.session_state.pop("suggested_rag_query")

    # Display past conversation history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("contexts"):
                with st.expander("📚 Rujukan Data Nutrisi (Katalog Komposisi Pangan)"):
                    for doc in msg["contexts"]:
                        st.markdown(f"- {doc}")

    # Quick suggestion chips for laypersons
    st.markdown("##### 💡 Contoh Pertanyaan Cepat (Klik untuk Bertanya):")
    chip_cols = st.columns(4)
    if chip_cols[0].button("🍗 Dada Ayam vs Paha", key="chip_ayam", use_container_width=True):
        st.session_state["suggested_rag_query"] = "Berapa kalori dan protein dalam dada ayam fillet?"
        st.rerun()
    if chip_cols[1].button("🍳 Sepiring Nasi Goreng", key="chip_nasgor", use_container_width=True):
        st.session_state["suggested_rag_query"] = "Berapa kalori sepiring nasi goreng telur?"
        st.rerun()
    if chip_cols[2].button("🥑 Lemak Baik Alpukat", key="chip_alpukat", use_container_width=True):
        st.session_state["suggested_rag_query"] = "Berapa kalori dan lemak sehat dalam satu buah alpukat?"
        st.rerun()
    if chip_cols[3].button("🥗 Camilan Rendah Gula", key="chip_camilan", use_container_width=True):
        st.session_state["suggested_rag_query"] = "Makanan apa yang tinggi serat dan rendah gula untuk diet?"
        st.rerun()

    # Query Input area
    col_input, col_clear = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Pertanyaan Anda:",
            placeholder="Contoh: Berapa kalori dan protein dalam dada ayam?",
            key="rag_query_input",
        )
    with col_clear:
        if st.button("Hapus Chat", key="clear_chat_button", help="Bersihkan riwayat percakapan"):
            st.session_state.chat_messages = []
            st.rerun()

    st.caption("💡 **Tips untuk Anda:** Anda bisa menanyakan nama makanan Indonesia, kandungan gizi makro, atau perbandingan antar makanan.")

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

    st.markdown("---")
    with st.expander("🥄 Kalkulator Takaran Porsi Rumah Tangga (URT) — Centong, Butir, Potong", expanded=False):
        st.write(
            "Ubah takaran 100 gram standar laboratorium menjadi porsi piring makan sehari-hari "
            "(misal: centong nasi, butir telur, potong ayam)."
        )
        c_urt_f, c_urt_p = st.columns([2, 2])
        with c_urt_f:
            urt_food_input = st.text_input(
                "Nama Makanan untuk Dihitung:",
                value="Nasi putih",
                key="urt_food_name_input",
                placeholder="Contoh: Nasi putih, Telur ayam, Dada ayam",
            )
        options = get_serving_options(urt_food_input)
        with c_urt_p:
            urt_choice = st.selectbox(
                "Pilih Takaran Porsi Sehari-hari:",
                options=[opt.label for opt in options],
                key="urt_serving_select",
            )
        selected_opt = next((opt for opt in options if opt.label == urt_choice), options[0])

        urt_matches = search_nutrition_keyword_ranked(urt_food_input, db_path=db_path, limit=1)
        if urt_matches:
            base_item = urt_matches[0]
            scaled = calculate_nutrition_for_serving(
                calories_100g=float(base_item["calories"]),
                proteins_100g=float(base_item["proteins"]),
                fat_100g=float(base_item["fat"]),
                carb_100g=float(base_item["carbohydrate"]),
                serving_grams=selected_opt.grams,
            )
            st.success(format_serving_summary(base_item["name"], selected_opt.label, scaled))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Energi", f"{scaled['calories']} kkal")
            m2.metric("Protein", f"{scaled['proteins']} g")
            m3.metric("Lemak", f"{scaled['fat']} g")
            m4.metric("Karbohidrat", f"{scaled['carbohydrate']} g")
        else:
            st.info(f"Ketik nama makanan yang terdaftar untuk menghitung takaran {selected_opt.label}.")


def _render_recipe_generator(
    llm_client: Any,
    model_name: str,
    db_path: str = "",
) -> None:
    """Render the ingredient-based healthy recipe generator tab with diet filters."""
    st.header("🍳 Masak Apa Hari Ini?")
    st.write(
        "Punya bahan di dapur dan bingung mau masak apa? Tulis bahan yang Anda miliki, "
        "dan NutriChatBot akan meracikkan ide hidangan sehat, lezat, dan bergizi seimbang."
    )

    if st.session_state.get("suggested_recipe"):
        st.session_state["recipe_ingredients"] = st.session_state.pop("suggested_recipe")

    st.markdown("##### 🥦 Pilihan Cepat Kombinasi Bahan:")
    r1, r2, r3 = st.columns(3)
    if r1.button("🍗 Ayam + Brokoli + Wortel", key="rec_ayam", use_container_width=True):
        st.session_state["suggested_recipe"] = "dada ayam fillet, brokoli, wortel, bawang putih"
        st.rerun()
    if r2.button("🌱 Tahu + Tempe + Bayam", key="rec_nabati", use_container_width=True):
        st.session_state["suggested_recipe"] = "tahu putih, tempe, bayam, bawang merah"
        st.rerun()
    if r3.button("🐟 Ikan Kembung + Tomat", key="rec_ikan", use_container_width=True):
        st.session_state["suggested_recipe"] = "ikan kembung, tomat merah, jeruk nipis, kemangi"
        st.rerun()

    ingredients = st.text_input(
        "Bahan-bahan yang Anda miliki (pisahkan koma):",
        placeholder="Contoh: ayam, brokoli, bawang putih",
        key="recipe_ingredients",
    )

    # Show authentic Indonesian regional recipes matching pantry items if database is available
    if db_path and ingredients.strip():
        pantry_list = [x.strip() for x in ingredients.split(",") if x.strip()]
        matched_recipes = find_recipes_by_pantry_ingredients(pantry_list, limit=3, db_path=db_path)
        if matched_recipes:
            st.markdown("##### 📖 Resep Nusantara Otentik yang Cocok dengan Bahan Anda:")
            for rec in matched_recipes:
                with st.expander(
                    f"🍲 {rec.name} ({rec.origin_region}) — ~{rec.calories_per_serving:.0f} kkal/porsi "
                    f"({rec.match_percentage:.0f}% Bahan Cocok)",
                    expanded=False,
                ):
                    st.caption(
                        f"⏱️ Waktu: {rec.prep_time_minutes} mnt | Porsi: {rec.servings} org | 🏷️ {rec.health_tags}"
                    )
                    st.write("**Bahan-Bahan:**")
                    for ing in rec.matched_ingredients:
                        st.markdown(f"- ✅ **{ing}** *(Tersedia)*")
                    for ing in rec.missing_ingredients:
                        st.markdown(f"- ⚪ {ing} *(Pelengkap/Bumbu)*")
                    st.write("**Langkah Memasak Otentik:**")
                    st.write(rec.instructions_text)
    diet_pref = st.selectbox(
        "Preferensi Diet / Kondisi Kesehatan:",
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

    diet_guides = {
        "Biasa (Tanpa Preferensi)": "Pola makan seimbang harian dengan bahan segar.",
        "Tinggi Protein": "Memprioritaskan lauk pembangun otot dan menjaga rasa kenyang lebih lama.",
        "Rendah Garam (Hipertensi)": "Membatasi garam dapur, kecap asin, dan bahan tinggi natrium.",
        "Rendah Gula (Diabetes)": "Bebas gula pasir atau pemanis sirup, ramah kestabilan gula darah.",
        "Vegetarian": "Bebas daging hewani (masih memperbolehkan telur dan susu).",
        "Vegan": "100% nabati murni (tanpa daging, telur, susu, atau madu).",
        "Bebas Gluten": "Bebas tepung gandum/terigu, aman untuk lambung sensitif/celiac.",
    }
    st.caption(f"ℹ️ **Panduan Pilihan:** {diet_guides.get(diet_pref, '')}")

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
    st.write(
        "Rencanakan pola makan sehat selama 7 hari sesuai target kalori harian Anda. "
        "Menu disusun praktis, lezat, dan mudah dimasak di rumah."
    )

    st.info(
        "💡 **Panduan Pemula:** Setiap orang memiliki kebutuhan kalori harian yang berbeda. "
        "Jika belum mengetahui kebutuhan kalori harian Anda, buka kalkulator di bawah ini "
        "agar rencana menu sesuai dengan postur dan aktivitas tubuh Anda."
    )

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
            "Tingkat Aktivitas Fisik Sehari-hari:",
            [
                "Sedentary (Jarang/Tidak Berolahraga - Banyak Duduk)",
                "Light (Olahraga Ringan 1-3 hari/minggu)",
                "Moderate (Olahraga Sedang 3-5 hari/minggu)",
                "Very Active (Olahraga Berat / Pekerja Fisik 6-7 hari)",
            ],
            key="bmr_activity",
        )
        calc_goal = st.selectbox(
            "Tujuan Berat Badan Anda:",
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

            bmi_badge = "🟢 Ideal" if "Normal" in plan["bmi_category"] else "🟡 Perlu Perhatian"
            st.success(
                f"🎉 **Hasil Analisis Tubuh Anda:**\n\n"
                f"- **Kebutuhan Kalori Target:** `{plan['target_calories']} kcal/hari`\n"
                f"- **Indeks Massa Tubuh (BMI):** `{plan['bmi']} ({plan['bmi_category']})` {bmi_badge}\n"
                f"- **BMR:** `{plan['bmr']} kcal` *(energi minimal saat istirahat)* | **TDEE:** `{plan['tdee']} kcal` *(total bakar harian)*"
            )

            st.markdown("##### 🍽️ Rekomendasi Porsi Makronutrisi Seimbang (Isi Piringku):")
            col_p, col_f, col_c = st.columns(3)
            col_p.metric("Target Protein (25%)", f"{plan['macros']['protein_g']}g")
            col_f.metric("Target Lemak Sehat (25%)", f"{plan['macros']['fat_g']}g")
            col_c.metric("Karbohidrat Kompleks (50%)", f"{plan['macros']['carb_g']}g")
            st.caption(
                "ℹ️ **Catatan:** Angka target kalori di atas otomatis digunakan pada kotak target di bawah."
            )

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
        "Bandingkan kandungan kalori dan zat gizi makro antara dua jenis makanan "
        "(data berbasis standar 100 gram porsi dapat dimakan)."
    )

    if st.session_state.get("suggested_f1"):
        st.session_state["compare_food1_input"] = st.session_state.pop("suggested_f1")
    if st.session_state.get("suggested_f2"):
        st.session_state["compare_food2_input"] = st.session_state.pop("suggested_f2")

    st.markdown("##### ⚖️ Contoh Perbandingan Populer:")
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("Tempe vs Tahu", key="preset_tempe_tahu", use_container_width=True):
        st.session_state["suggested_f1"] = "Tempe"
        st.session_state["suggested_f2"] = "Tahu"
        st.rerun()
    if p2.button("Beras Putih vs Merah", key="preset_beras", use_container_width=True):
        st.session_state["suggested_f1"] = "Beras giling"
        st.session_state["suggested_f2"] = "Beras merah"
        st.rerun()
    if p3.button("Dada Ayam vs Sapi", key="preset_daging", use_container_width=True):
        st.session_state["suggested_f1"] = "Dada ayam"
        st.session_state["suggested_f2"] = "Daging sapi"
        st.rerun()
    if p4.button("Telur vs Ikan Kembung", key="preset_ikan", use_container_width=True):
        st.session_state["suggested_f1"] = "Telur ayam"
        st.session_state["suggested_f2"] = "Ikan kembung"
        st.rerun()

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

        # Quick Takeaway Highlights for Laypersons
        cal_diff = abs(float(f1["calories"]) - float(f2["calories"]))
        prot_diff = abs(float(f1["proteins"]) - float(f2["proteins"]))
        lower_cal = f1["name"] if float(f1["calories"]) <= float(f2["calories"]) else f2["name"]
        higher_prot = f1["name"] if float(f1["proteins"]) >= float(f2["proteins"]) else f2["name"]

        st.markdown("##### 💡 Analisis Cepat untuk Anda:")
        t1, t2 = st.columns(2)
        with t1:
            st.info(
                f"🥗 **Lebih Rendah Kalori:** `{lower_cal}` "
                f"(selisih ~{cal_diff:.0f} kkal per 100g). Cocok untuk program defisit kalori."
            )
        with t2:
            st.info(
                f"🍗 **Lebih Tinggi Protein:** `{higher_prot}` "
                f"(selisih ~{prot_diff:.1f}g per 100g). Membantu menjaga rasa kenyang lebih lama."
            )

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

        # Interactive Household Serving Size (URT) Comparison
        with st.expander("🥄 Hitung Perbandingan Berdasarkan Takaran Porsi Piring (URT)", expanded=False):
            st.write("Bandingkan makanan berdasarkan takaran porsi lazim santap (bukan hanya 100 gram).")
            opts1 = get_serving_options(f1["name"])
            opts2 = get_serving_options(f2["name"])
            cu1, cu2 = st.columns(2)
            with cu1:
                sel1_label = st.selectbox(
                    f"Takaran {f1['name']}:",
                    [opt.label for opt in opts1],
                    key="comp_urt1_sel",
                )
                opt1 = next((o for o in opts1 if o.label == sel1_label), opts1[0])
                scaled1 = calculate_nutrition_for_serving(
                    float(f1["calories"]), float(f1["proteins"]), float(f1["fat"]), float(f1["carbohydrate"]), opt1.grams
                )
            with cu2:
                sel2_label = st.selectbox(
                    f"Takaran {f2['name']}:",
                    [opt.label for opt in opts2],
                    key="comp_urt2_sel",
                )
                opt2 = next((o for o in opts2 if o.label == sel2_label), opts2[0])
                scaled2 = calculate_nutrition_for_serving(
                    float(f2["calories"]), float(f2["proteins"]), float(f2["fat"]), float(f2["carbohydrate"]), opt2.grams
                )

            cu_m1, cu_m2 = st.columns(2)
            with cu_m1:
                st.write(f"**{f1['name']} ({opt1.label}):**")
                st.write(f"🔥 {scaled1['calories']} kkal | 🍗 {scaled1['proteins']}g Prot | 🥑 {scaled1['fat']}g Lemak | 🍚 {scaled1['carbohydrate']}g Karbo")
            with cu_m2:
                st.write(f"**{f2['name']} ({opt2.label}):**")
                st.write(f"🔥 {scaled2['calories']} kkal | 🍗 {scaled2['proteins']}g Prot | 🥑 {scaled2['fat']}g Lemak | 🍚 {scaled2['carbohydrate']}g Karbo")


def _render_packaged_foods(db_path: str) -> None:
    """Render search and Nutri-Score explorer for Indonesian packaged foods (FMCG)."""
    st.header("📦 Produk Kemasan & Nutri-Score (FMCG)")
    st.write(
        "Cek kandungan gizi produk kemasan supermarket & minimarket Indonesia berdasarkan label resmi BPOM. "
        "Dilengkapi sistem pemeringkatan **Nutri-Score** (A = Paling Sehat hingga E = Konsumsi Dibatasi)."
    )

    categories = ["Semua Kategori"] + get_packaged_categories(db_path)
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        pkg_query = st.text_input(
            "Cari Produk atau Merek:",
            placeholder="Contoh: Indomie, Ultra Milk, Bear Brand, Roma, Quaker",
            key="pkg_search_query",
        )
    with c2:
        selected_cat = st.selectbox("Kategori Produk:", categories, key="pkg_cat_filter")
    with c3:
        selected_score = st.selectbox(
            "Filter Nutri-Score:",
            ["Semua Peringkat", "A (Paling Sehat)", "B (Sehat)", "C (Sedang)", "D (Tinggi Gula/Lemak)", "E (Konsumsi Dibatasi)"],
            key="pkg_score_filter",
        )

    score_char = ""
    if "A" in selected_score:
        score_char = "A"
    elif "B" in selected_score:
        score_char = "B"
    elif "C" in selected_score:
        score_char = "C"
    elif "D" in selected_score:
        score_char = "D"
    elif "E" in selected_score:
        score_char = "E"

    cat_filter = "" if selected_cat == "Semua Kategori" else selected_cat

    results = search_packaged_foods(
        query=pkg_query,
        category=cat_filter,
        nutri_score=score_char,
        limit=20,
        db_path=db_path,
    )

    st.markdown(f"**Menemukan {len(results)} produk kemasan:**")
    if not results:
        st.info("Tidak ada produk kemasan yang cocok dengan kata kunci atau filter yang dipilih.")
        return

    for item in results:
        score = str(item.get("nutri_score", "C")).upper()
        badge_class = f"nutri-{score.lower()}"
        with st.container():
            st.markdown(
                f"""
                <div class="fmcg-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">{item['name']}</span>
                        <span class="nutri-score-pill {badge_class}">Nutri-Score {score}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.88rem; margin-bottom: 8px;">
                        🏷️ Merek: <b>{item['brand']}</b> | Kategori: <b>{item['category']}</b> | Takaran Saji: <b>{item['serving_unit']}</b>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.92rem; margin-bottom: 8px;">
                        {item['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Energi / Saji", f"{item['calories']:.0f} kkal")
            col2.metric("Protein", f"{item['proteins']:.1f} g")
            col3.metric("Gula", f"{item['sugars']:.1f} g")
            col4.metric("Lemak Jenuh", f"{item['saturated_fat']:.1f} g")
            col5.metric("Natrium (Garam)", f"{item['sodium_mg']:.0f} mg")

            # Health warnings for extreme nutrients
            warnings = []
            if float(item["sugars"]) >= 15.0:
                warnings.append("⚠️ **Tinggi Gula:** Mengandung ≥15g gula per saji, batasi jika mengontrol berat badan/diabetes.")
            if float(item["sodium_mg"]) >= 800.0:
                warnings.append("⚠️ **Tinggi Garam/Natrium:** Mengandung ≥800mg natrium per saji, waspadai jika memiliki hipertensi.")
            if float(item["saturated_fat"]) >= 5.0:
                warnings.append("⚠️ **Tinggi Lemak Jenuh:** Mengandung ≥5g lemak jenuh per saji.")
            if warnings:
                for w in warnings:
                    st.caption(w)
            st.markdown("---")


def _render_healthy_alternatives(llm_client: Any, model_name: str) -> None:
    """Render the healthy ingredient substitution tab."""
    st.header("🥗 Alternatif Sehat")
    st.write(
        "Ingin hidup lebih sehat tetapi bingung mencari pengganti bahan masakan favorit? "
        "Tulis bahan makanan yang ingin Anda ganti, dan temukan alternatif yang lebih ramah tubuh."
    )

    if st.session_state.get("suggested_alt"):
        st.session_state["healthy_alternative_input"] = st.session_state.pop("suggested_alt")

    st.markdown("##### 🔄 Bahan yang Sering Ingin Diganti:")
    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Gula Pasir", key="alt_gula", use_container_width=True):
        st.session_state["suggested_alt"] = "gula pasir"
        st.rerun()
    if a2.button("Minyak Kelapa Sawit", key="alt_minyak", use_container_width=True):
        st.session_state["suggested_alt"] = "minyak goreng kelapa sawit"
        st.rerun()
    if a3.button("Santan Kental", key="alt_santan", use_container_width=True):
        st.session_state["suggested_alt"] = "santan kental"
        st.rerun()
    if a4.button("Tepung Terigu", key="alt_terigu", use_container_width=True):
        st.session_state["suggested_alt"] = "tepung terigu"
        st.rerun()

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
    st.header("⚙️ Pengaturan & Indeks Data (Admin)")
    st.warning(
        "⚠️ **Area Khusus Administrator:** Halaman ini digunakan untuk memuat ulang dataset pangan "
        "ke database SQLite & ChromaDB. Pengguna umum tidak perlu menjalankan fungsi ini untuk konsultasi gizi."
    )
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
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_custom_styles()

    st.title("NutriChatBot: Asisten Gizi & Resep Sehat 🍎🥕")
    st.markdown(
        "Teman cerdas Anda untuk berkonsultasi seputar nutrisi makanan, ide masakan sehat, "
        "kalkulator kalori personal, dan perbandingan gizi praktis."
    )

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
    packaged_count = get_packaged_nutrition_count(settings.db_path)
    recipes_count = get_recipes_count(settings.db_path)
    chroma_count = get_collection_count(chroma_client, embedding_fn)

    menu_options = [
        "Tanya Gizi",
        "Masak Apa Hari Ini?",
        "Perencana Menu Mingguan",
        "Bandingkan Makanan",
        "Produk Kemasan & Nutri-Score",
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
        st.markdown("## 🥗 NutriChatBot")
        st.caption("Panduan Gizi & Hidup Sehat Nusantara")

        st.markdown("---")
        st.markdown("#### 🧭 Menu Layanan")
        menu_selection = st.radio(
            "Pilih Opsi",
            menu_options,
            index=default_idx,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("#### 🟢 Status Asisten")
        st.caption("✅ Terhubung & Siap Melayani")
        st.markdown(f"📚 **Katalog Pangan DKBM:** `{sqlite_count} makanan`")
        st.markdown(f"📦 **Produk Kemasan FMCG:** `{packaged_count} produk`")
        st.markdown(f"🍲 **Resep Otentik:** `{recipes_count} masakan`")
        st.markdown("🏛️ **Rujukan:** DKBM Kemenkes RI & BPOM")

        if sqlite_count == 0 or chroma_count == 0:
            st.warning("Data belum diindeks. Buka menu 'Index Data' untuk memuat dataset nutrisi.")

        st.markdown("---")
        with st.expander("🛠️ Detail Sistem (Teknis)", expanded=False):
            st.write(f"**Model LLM:** `{settings.model_llm}`")
            st.write(f"**Model Embedding:** `{settings.model_embedding}`")
            st.write(f"**Endpoint:** `{settings.base_url}`")
            st.write(f"**Database SQLite:** {sqlite_count} item pangan")
            st.write(f"**Produk Kemasan:** {packaged_count} produk")
            st.write(f"**Resep Nusantara:** {recipes_count} resep")
            st.write(f"**Index Vektor:** {chroma_count} dokumen")

    if menu_selection == "Tanya Gizi":
        _render_tanya_gizi(
            llm_client=llm_client,
            chroma_client=chroma_client,
            embedding_fn=embedding_fn,
            db_path=str(settings.db_path),
            model_name=settings.model_llm,
        )
    elif menu_selection == "Masak Apa Hari Ini?":
        _render_recipe_generator(
            llm_client=llm_client,
            model_name=settings.model_llm,
            db_path=str(settings.db_path),
        )
    elif menu_selection == "Perencana Menu Mingguan":
        _render_weekly_planner(llm_client=llm_client, model_name=settings.model_llm)
    elif menu_selection == "Bandingkan Makanan":
        _render_food_comparison(db_path=str(settings.db_path))
    elif menu_selection == "Produk Kemasan & Nutri-Score":
        _render_packaged_foods(db_path=str(settings.db_path))
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
