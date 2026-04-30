import streamlit as st
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
    # APPLY PREPROCESSING (IMPUTATION ONLY)
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

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        use_validation = st.checkbox("Use Validation Set")

        val_size = 0.2 if use_validation else 0

        # ======================
        # TIME-BASED OPTION
        # ======================
        if method == "time_based":
            date_col = st.selectbox("Select Date Column", df.columns)

            df = df.sort_values(by=date_col)

            split_idx = int(len(df) * (1 - test_size))

            train = df.iloc[:split_idx]
            test = df.iloc[split_idx:]
            val = None

        else:
            stratify_col = df[target] if method == "stratified" else None

            if use_validation:
                train_val, test = train_test_split(
                    df,
                    test_size=test_size,
                    stratify=stratify_col,
                    random_state=42
                )

                val_ratio = val_size / (1 - test_size)

                stratify_val = train_val[target] if method == "stratified" else None

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
        # RESULT
        # ======================
        st.subheader("📊 Split Result")

        st.write("Train:", train.shape)
        st.write("Test:", test.shape)

        if val is not None:
            st.write("Validation:", val.shape)

        # ======================
        # DATA PREVIEW (NEW)
        # ======================
        st.subheader("🔍 Data Preview")

        with st.expander("Train Data"):
            st.dataframe(train.head(100), width='stretch')

        with st.expander("Test Data"):
            st.dataframe(test.head(100), width='stretch')

        if val is not None:
            with st.expander("Validation Data"):
                st.dataframe(val.head(100), width='stretch')

        # ======================
        # SAVE
        # ======================
        if st.button("💾 Save Split"):
            save_split(project_id, train, test, val, method)
            st.success("Split saved!")
            st.session_state["resplit"] = False
            st.rerun()
