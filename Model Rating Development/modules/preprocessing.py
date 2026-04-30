import streamlit as st
import pandas as pd

from database.crud import load_dataset, save_preprocessing, load_preprocessing

def detect_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    else:
        return "categorical"


def run(project_id):
    st.header("⚙️ Preprocessing")

    # ======================
    # LOAD DATASET
    # ======================
    df, file_name = load_dataset(project_id)

    if df is None:
        st.warning("Please upload dataset first")
        return

    st.success(f"Dataset: {file_name}")

    # ======================
    # LOAD EXISTING CONFIG
    # ======================
    config = load_preprocessing(project_id)

    # ======================
    # TARGET VARIABLE
    # ======================
    st.subheader("🎯 Target Variable")

    target = st.selectbox(
        "Select Target",
        df.columns,
        index=df.columns.get_loc(config["target"]) if config else 0
    )

    # ======================
    # FEATURE SELECTION
    # ======================
    st.subheader("📊 Feature Selection")

    default_features = config["features"] if config else []

    features = st.multiselect(
        "Select Features",
        df.columns.drop(target),
        default=default_features
    )

    # ======================
    # TYPE DETECTION + EDITABLE
    # ======================
    st.subheader("🔍 Variable Types")

    type_dict = {}

    for col in features:
        detected_type = detect_type(df[col])

        # Tambahan: deteksi datetime sederhana
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            detected_type = "datetime"

        options = ["numeric", "categorical", "datetime"]

        selected_type = st.selectbox(
            f"Type for {col}",
            options,
            index=options.index(detected_type) if detected_type in options else 1,
            key=f"type_{col}"
        )

        type_dict[col] = selected_type

    type_df = pd.DataFrame({
        "Variable": list(type_dict.keys()),
        "Type": list(type_dict.values())
    })

    st.dataframe(type_df, width='stretch')

    # ======================
    # MISSING SUMMARY
    # ======================
    st.subheader("📉 Missing Value Summary")

    missing_df = pd.DataFrame({
        "Variable": df.columns,
        "Missing Count": df.isna().sum().values,
    })

    missing_df["Missing %"] = (missing_df["Missing Count"] / len(df)) * 100
    missing_df = missing_df.sort_values(by="Missing %", ascending=False)

    st.dataframe(missing_df, width='stretch')

    # ======================
    # FILTER VARIABLE YANG ADA MISSING
    # ======================
    missing_vars = missing_df[missing_df["Missing Count"] > 0]["Variable"].tolist()

    # ======================
    # IMPUTATION SECTION
    # ======================
    st.subheader("🧹 Imputation Configuration")

    imputation_rules = {}

    if len(missing_vars) == 0:
        st.success("No missing values found 🎉")

    else:
        for col in missing_vars:
            col_type = detect_type(df[col])
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df) * 100

            st.markdown(f"### {col}")
            st.caption(f"Missing: {missing_count} ({missing_pct:.2f}%)")

            # ======================
            # NUMERIC
            # ======================
            if col_type == "numeric":
                method = st.selectbox(
                    f"Imputation method for {col}",
                    ["mean", "median", "mode", "manual"],
                    key=f"{col}_method"
                )

                if method == "manual":
                    value = st.number_input(
                        f"Manual value for {col}",
                        key=f"{col}_value"
                    )
                    imputation_rules[col] = {
                        "method": "manual",
                        "value": value
                    }
                else:
                    imputation_rules[col] = {
                        "method": method
                    }

            # ======================
            # CATEGORICAL
            # ======================
            else:
                method = st.selectbox(
                    f"Imputation method for {col}",
                    ["mode", "manual"],
                    key=f"{col}_cat_method"
                )

                if method == "manual":
                    value = st.text_input(
                        f"Manual value for {col}",
                        key=f"{col}_cat_value"
                    )
                    imputation_rules[col] = {
                        "method": "manual",
                        "value": value
                    }
                else:
                    imputation_rules[col] = {
                        "method": "mode"
                    }

    # ======================
    # SAVE BUTTON
    # ======================
    if st.button("💾 Save Preprocessing"):
        save_preprocessing(project_id, target, features, imputation_rules)
        st.success("Preprocessing saved!")
