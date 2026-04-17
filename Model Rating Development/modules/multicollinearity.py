import streamlit as st
import pandas as pd

from database.crud import load_split, load_preprocessing, load_binning, save_model_dataset
from utils.binning import apply_binning
from utils.woe import calculate_woe_iv
from utils.vif import calculate_vif
from utils.transform import apply_transformation  # 🔥 NEW


def run(project_id):
    st.header("🔍 Multicollinearity (VIF)")

    # ======================
    # LOAD DATA
    # ======================
    split = load_split(project_id)
    config = load_preprocessing(project_id)
    binning_rules = load_binning(project_id)

    if split is None or config is None or binning_rules is None:
        st.warning("Complete previous steps first")
        return

    df = split["train"].copy()
    target = config["target"]
    features = config["features"]

    # ======================
    # VARIABLE SELECTION (NEW)
    # ======================
    st.subheader("📌 Select Variables for VIF")

    selected_features = st.multiselect(
        "Choose variables",
        features,
        default=features
    )

    # ======================
    # 🔥 APPLY TRANSFORMATION (FIX)
    # ======================
    df = apply_transformation(df, binning_rules)

    st.success("Transformation applied")

    # ======================
    # APPLY BINNING
    # ======================
    df_binned = apply_binning(df, binning_rules)

    st.success("Binning applied")

    # ======================
    # WOE TRANSFORMATION (ROBUST - MERGE BASED)
    # ======================
    df_woe = pd.DataFrame(index=df.index)  # 🔥 ensure index align
    warning_cols = []

    for col in selected_features:
        try:
            woe_table, _ = calculate_woe_iv(df_binned, col, target)

            # 🔥 IMPORTANT: gunakan merge (bukan map)
            temp = df_binned[[col]].copy()
            temp["bin"] = temp[col]

            woe_table_copy = woe_table.copy()
            woe_table_copy["bin"] = woe_table_copy["bin"]

            merged = temp.merge(
                woe_table_copy[["bin", "woe"]],
                on="bin",
                how="left"
            )

            df_woe[col] = merged["woe"].values  # 🔥 FIX alignment

            # ======================
            # VALIDATION
            # ======================
            missing_ratio = df_woe[col].isna().mean()

            if missing_ratio > 0:
                warning_cols.append((col, missing_ratio))

            # fill missing
            df_woe[col] = df_woe[col].fillna(0)

        except Exception as e:
            st.error(f"Error processing {col}: {e}")

    st.success("WOE transformation completed")

    # ======================
    # VALIDATION DISPLAY
    # ======================
    st.subheader("🔍 WOE Validation")

    validation_df = pd.DataFrame({
        "variable": df_woe.columns,
        "missing_ratio": df_woe.isna().mean(),
        "unique_values": df_woe.nunique(),
        "mean": df_woe.mean(),
        "std": df_woe.std()
    })

    st.dataframe(validation_df, use_container_width=True)

    # warning detail
    if warning_cols:
        st.warning("Some variables have missing after WOE mapping:")

        for col, ratio in warning_cols:
            st.write(f"- {col}: {ratio:.2%} missing → filled with 0")

            if ratio > 0.2:
                st.error(f"{col} mapping is unreliable (>20% missing)")

    # ======================
    # WOE DATA PREVIEW
    # ======================
    st.subheader("📊 WOE Transformed Data")

    n_rows = st.slider("Number of rows to display", 5, 100, 20)

    st.dataframe(df_woe.head(n_rows), use_container_width=True)

    # ======================
    # VALIDATE NON-EMPTY
    # ======================
    valid_cols = [
        col for col in df_woe.columns
        if df_woe[col].notna().sum() > 0
    ]

    df_woe = df_woe[valid_cols]

    if df_woe.empty:
        st.error("No valid variables for VIF calculation")
        return

    # ======================
    # VIF CALCULATION
    # ======================
    vif_df = calculate_vif(df_woe)
    vif_df = vif_df.sort_values(by="vif", ascending=False)

    st.subheader("📊 VIF Result")
    st.dataframe(vif_df, use_container_width=True)

    # ======================
    # RECOMMENDATION
    # ======================
    st.subheader("🧠 Recommendation")

    high_vif = vif_df[vif_df["vif"] > 10]["variable"].tolist()
    medium_vif = vif_df[(vif_df["vif"] > 5) & (vif_df["vif"] <= 10)]["variable"].tolist()
    low_vif = vif_df[vif_df["vif"] <= 5]["variable"].tolist()

    st.write("### ❌ High Multicollinearity (VIF > 10)")
    st.write(high_vif)

    st.write("### ⚠️ Medium Multicollinearity (5–10)")
    st.write(medium_vif)

    st.write("### ✅ Safe Variables (VIF ≤ 5)")
    st.write(low_vif)

    st.caption("⚠️ Final decision remains with user")

    # ======================
    # MANUAL SELECTION
    # ======================
    st.subheader("✏️ Select Variables for Model")

    default_vars = low_vif if low_vif else vif_df["variable"].tolist()

    selected_vars = st.multiselect(
        "Choose variables to KEEP",
        options=vif_df["variable"].tolist(),
        default=default_vars
    )

    st.write("Selected variables:")
    st.write(selected_vars)

    # ======================
    # SAVE FINAL FEATURES + WOE DATASET
    # ======================
    if st.button("💾 Save Selected Variables"):

        selected_df_woe = df_woe[selected_vars]

        # ======================
        # SAVE TO SESSION
        # ======================
        st.session_state["final_features"] = selected_vars
        st.session_state["df_woe"] = selected_df_woe

        # ======================
        # SAVE TO DATABASE (PERSIST)
        # ======================
        try:
            save_model_dataset(
                project_id,
                selected_df_woe,
                selected_vars
            )

            st.success("Variables and WOE dataset saved (persistent)!")

        except Exception as e:
            st.error(f"Failed to save dataset: {e}")
