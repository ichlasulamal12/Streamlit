import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split

from database.crud import load_dataset, load_preprocessing, save_split, load_split
from utils.helpers import apply_imputation


def run(project_id):
    st.header("✂️ Split Data")

    # ======================
    # LOAD DATA
    # ======================
    df, _ = load_dataset(project_id)
    config = load_preprocessing(project_id)

    if df is None or config is None:
        st.warning("Complete Input Data & Preprocessing first")
        return

    # ======================
    # APPLY PREPROCESSING
    # ======================
    df = apply_imputation(df, config["imputation_rules"])

    target = config["target"]

    # ======================
    # CHECK EXISTING SPLIT
    # ======================
    existing = load_split(project_id)

    if existing:
        st.success(f"Split already exists ({existing['method']})")
        st.write("Train shape:", existing["train"].shape)
        st.write("Test shape:", existing["test"].shape)

        if existing["val"] is not None:
            st.write("Validation shape:", existing["val"].shape)

        if st.button("🔄 Re-split Data"):
            st.session_state["resplit"] = True
    else:
        st.session_state["resplit"] = True

    # ======================
    # SPLIT CONFIG
    # ======================
    if st.session_state.get("resplit", False):

        st.subheader("⚙️ Split Configuration")

        method = st.selectbox(
            "Split Method",
            ["random", "stratified", "time_based"]
        )

        use_validation = st.checkbox("Use Validation Set")

        val_size = 0.2 if use_validation else 0

        # =====================================================
        # RANDOM / STRATIFIED
        # =====================================================
        if method != "time_based":

            test_size = st.slider(
                "Test Size",
                min_value=0.10,
                max_value=0.50,
                value=0.20,
                step=0.05
            )

            stratify_col = df[target] if method == "stratified" else None

            if use_validation:

                train_val, test = train_test_split(
                    df,
                    test_size=test_size,
                    stratify=stratify_col,
                    random_state=42
                )

                val_ratio = val_size / (1 - test_size)

                stratify_val = (
                    train_val[target]
                    if method == "stratified"
                    else None
                )

                train, val = train_test_split(
                    train_val,
                    test_size=val_ratio,
                    stratify=stratify_val,
                    random_state=42
                )

            else:

                train, test = train_test_split(
                    df,
                    test_size=test_size,
                    stratify=stratify_col,
                    random_state=42
                )

                val = None

        # ======================
        # TIME BASED
        # ======================
        else:

            date_col = st.selectbox(
                "Date Column",
                df.columns
            )

            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception:
                st.error(
                    f"Column '{date_col}' is not a valid date column. "
                    "Please select another column."
                )
                st.stop()

            df = df.sort_values(date_col)

            split_date = st.date_input(
                "Split Date",
                value=df[date_col].max().date(),
                min_value=df[date_col].min().date(),
                max_value=df[date_col].max().date()
            )

            split_date = pd.Timestamp(split_date)

            train = df[df[date_col] < split_date]
            test = df[df[date_col] >= split_date]
            val = None

        # ======================
        # RESULT
        # ======================
        st.subheader("📊 Split Result")

        total = len(df)

        st.write(
            f"**Train :** {train.shape} ({len(train)/total:.1%})"
        )

        st.write(
            f"**Test :** {test.shape} ({len(test)/total:.1%})"
        )

        if val is not None:
            st.write(
                f"**Validation :** {val.shape} ({len(val)/total:.1%})"
            )

        # ======================
        # DATA PREVIEW
        # ======================
        st.subheader("🔍 Data Preview")

        with st.expander("Train Data"):
            st.dataframe(train.head(100), width="stretch")

        with st.expander("Test Data"):
            st.dataframe(test.head(100), width="stretch")

        if val is not None:
            with st.expander("Validation Data"):
                st.dataframe(val.head(100), width="stretch")

        # ======================
        # SAVE
        # ======================
        if st.button("💾 Save Split"):

            save_split(
                project_id=project_id,
                train=train,
                test=test,
                val=val,
                method=method
            )

            st.success("Split saved!")
            st.session_state["resplit"] = False
            st.rerun()
