import streamlit as st
import pandas as pd

from database.crud import (
    load_split,
    load_preprocessing,
    load_binning,
    load_model_dataset,
    save_model_dataset
)

from utils.binning import apply_binning
from utils.woe import calculate_woe_iv
from utils.transform import apply_transformation


# ======================
# COLOR IV
# ======================
def color_iv(val):
    if val < 0.02:
        return "background-color: #ffcccc"
    elif val < 0.1:
        return "background-color: #fff2cc"
    else:
        return "background-color: #ccffcc"


# ======================
# MAIN
# ======================
def run(project_id):

    st.header("📈 WOE & IV")

    # ======================
    # LOAD DATA
    # ======================
    split = load_split(project_id)
    config = load_preprocessing(project_id)
    binning_rules = load_binning(project_id)
    existing_model_data = load_model_dataset(project_id)

    if split is None or config is None or binning_rules is None:
        st.warning("Complete previous steps first")
        return

    df = split["train"].copy()
    target = config["target"]
    features = config["features"]

    # ======================
    # APPLY TRANSFORMATION
    # ======================
    df = apply_transformation(df, binning_rules)
    st.success("Transformation applied")

    # ======================
    # APPLY BINNING
    # ======================
    df_binned = apply_binning(df, binning_rules)
    st.success("Binning applied")

    iv_summary = []
    woe_results_all = []

    alpha = st.slider("Smoothing (alpha)", 0.1, 2.0, 0.5)

    # ======================
    # LOOP FEATURES
    # ======================
    for col in features:

        st.subheader(f"🔹 {col}")

        try:
            woe_table, iv = calculate_woe_iv(df_binned, col, target, alpha)

            # ======================
            # SORT
            # ======================
            if binning_rules[col]["type"] == "numeric":
                woe_table = woe_table.sort_values(by="bin")
            else:
                woe_table = woe_table.sort_values(by="woe", ascending=False)

            st.write(f"IV: {iv:.4f}")
            st.dataframe(woe_table, use_container_width=True)

            iv_summary.append({
                "variable": col,
                "iv": iv
            })

            # ======================
            # STANDARDIZE COLUMN NAME
            # ======================
            woe_table["variabel"] = col

            if "bin" in woe_table.columns:
                woe_table = woe_table.rename(columns={"bin": "kategori"})

            # 🔥 WAJIB INI
            woe_table["kategori"] = woe_table["kategori"].astype(str)

            woe_results_all.append(
                woe_table[["variabel", "kategori", "woe"]]
            )

        except Exception as e:
            st.error(f"Error processing {col}: {e}")

    # ======================
    # SAVE WOE RESULT
    # ======================
    if len(woe_results_all) > 0:

        woe_result = pd.concat(woe_results_all, ignore_index=True)

        # ======================
        # MERGE DENGAN DATA EXISTING (JANGAN HILANGKAN DATA LAIN)
        # ======================
        df_woe_existing = None
        if existing_model_data is not None:
            df_woe_existing = existing_model_data.get("df_woe")

        save_model_dataset(
            project_id=project_id,
            df_woe=df_woe_existing,  # 🔥 jangan overwrite
            features=features,
            woe_result=woe_result
        )

        st.success("WOE result saved to database")

        st.subheader("📦 WOE SOURCE (Saved)")
        st.dataframe(woe_result.head(100), use_container_width=True)

    else:
        st.warning("No WOE result generated")

    # ======================
    # IV SUMMARY
    # ======================
    st.subheader("📊 IV Summary")

    iv_df = pd.DataFrame(iv_summary)

    if not iv_df.empty:
        iv_df = iv_df.sort_values(by="iv", ascending=False)

        st.dataframe(
            iv_df.style.map(color_iv, subset=["iv"]),
            use_container_width=True
        )
    else:
        st.warning("No IV calculated")

    # ======================
    # INTERPRETATION
    # ======================
    st.write("### IV Interpretation")

    st.write("""
    - less than 0.02 → Not predictive  
    - 0.02–0.1 → Weak  
    - 0.1–0.3 → Medium  
    - more than 0.3 → Strong  
    """)

    # ======================
    # RECOMMENDATION
    # ======================
    st.subheader("🧠 Variable Recommendation")

    def categorize_iv(iv):
        if iv < 0.02:
            return "Not Predictive"
        elif iv < 0.1:
            return "Weak"
        elif iv < 0.3:
            return "Medium"
        else:
            return "Strong"

    if not iv_df.empty:

        iv_df["category"] = iv_df["iv"].apply(categorize_iv)

        recommended_keep = iv_df[iv_df["iv"] >= 0.1]["variable"].tolist()
        recommended_drop = iv_df[iv_df["iv"] < 0.02]["variable"].tolist()

        st.write("### ✅ Recommended Variables (Keep)")
        st.write(recommended_keep)

        st.write("### ❌ Suggested to Drop")
        st.write(recommended_drop)

    else:
        recommended_keep = []
        recommended_drop = []

    st.caption("⚠️ Final decision remains with user")

    # ======================
    # MANUAL SELECTION
    # ======================
    st.subheader("✏️ Manual Variable Selection")

    selected_vars = st.multiselect(
        "Select variables to KEEP",
        options=iv_df["variable"].tolist() if not iv_df.empty else [],
        default=recommended_keep
    )

    st.write("Selected variables:")
    st.write(selected_vars)

    # ======================
    # SAVE SELECTION
    # ======================
    if st.button("💾 Save Variable Selection"):

        st.session_state["selected_features"] = selected_vars

        st.success("Selected variables saved!")
