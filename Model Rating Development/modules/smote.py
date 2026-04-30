import streamlit as st
import pandas as pd

from imblearn.combine import SMOTETomek
from database.crud import load_split, load_preprocessing, load_model_dataset, save_model_dataset


def run(project_id):
    st.header("⚖️ SMOTE (Imbalance Handling)")

    # ======================
    # LOAD DATA
    # ======================
    split = load_split(project_id)
    config = load_preprocessing(project_id)

    if split is None or config is None:
        st.warning("Complete previous steps first")
        return

    # ======================
    # LOAD FROM SESSION / DB
    # ======================
    if "df_woe" in st.session_state and "final_features" in st.session_state:
        df_woe = st.session_state["df_woe"]
        selected_features = st.session_state["final_features"]

    else:
        model_data = load_model_dataset(project_id)

        if model_data is None:
            st.warning("Run VIF step first")
            return

        df_woe = model_data["df_woe"]
        selected_features = model_data["features"]

        # sync ke session
        st.session_state["df_woe"] = df_woe
        st.session_state["final_features"] = selected_features

    target = config["target"]

    df_train = split["train"].copy()
    valid_features = [col for col in selected_features if col in df_woe.columns]
    missing_features = list(set(selected_features) - set(valid_features))

    if missing_features:
        st.warning(f"Missing features dropped: {missing_features}")

    X = df_woe[valid_features].reset_index(drop=True)
    y = df_train[target].reset_index(drop=True)

    # ======================
    # BASIC INFO
    # ======================
    st.subheader("📌 Dataset Info")

    st.write("Feature shape:", X.shape)
    st.write("Target shape:", y.shape)

    # ======================
    # DATA PREVIEW (ORIGINAL)
    # ======================
    with st.expander("🔍 View Original Dataset"):
        preview_df = X.copy()
        preview_df[target] = y
        st.dataframe(preview_df.head(100), width='stretch')

    # ======================
    # ORIGINAL DISTRIBUTION
    # ======================
    st.subheader("📊 Original Target Distribution")

    original_dist = y.value_counts().rename("count").to_frame()
    original_dist["ratio"] = original_dist["count"] / len(y)

    st.dataframe(original_dist, width='stretch')

    # ======================
    # OPTION
    # ======================
    st.subheader("⚙️ SMOTE Configuration")

    use_smote = st.checkbox("Apply SMOTETomek")

    sampling_strategy = st.text_input(
        "Sampling Strategy (optional)",
        placeholder="e.g. 0.5 (minority = 50% of majority)"
    )

    # ======================
    # APPLY BUTTON
    # ======================
    if st.button("🚀 Prepare Dataset for Modelling"):

        # ======================
        # NO SMOTE
        # ======================
        if not use_smote:

            st.info("Using original dataset (no SMOTE)")

            st.session_state["X_model"] = X
            st.session_state["y_model"] = y
            st.session_state["use_smote"] = False

            st.session_state["model_data_info"] = {
                "source": "original",
                "rows": len(y),
                "features": X.shape[1]
            }

            # 🔥 SAVE TO DB
            save_model_dataset(
                project_id,
                df_woe,
                selected_features,
                source="original"
            )

            st.success("Dataset ready for modelling!")

            return

        # ======================
        # APPLY SMOTE
        # ======================
        try:
            st.write("Running SMOTETomek...")

            if sampling_strategy:
                sampling_strategy = float(sampling_strategy)
                smote = SMOTETomek(
                    sampling_strategy=sampling_strategy,
                    random_state=42
                )
            else:
                smote = SMOTETomek(random_state=42)

            X_resampled, y_resampled = smote.fit_resample(X, y)

            st.success("SMOTE applied successfully")

            # ======================
            # NEW DISTRIBUTION
            # ======================
            st.subheader("📊 New Target Distribution")

            new_dist = pd.Series(y_resampled).value_counts().rename("count").to_frame()
            new_dist["ratio"] = new_dist["count"] / len(y_resampled)

            st.dataframe(new_dist, width='stretch')

            # ======================
            # SHAPE INFO
            # ======================
            st.subheader("📌 New Dataset Info")

            st.write("New Feature shape:", X_resampled.shape)
            st.write("New Target shape:", y_resampled.shape)

            # ======================
            # DATA PREVIEW (SMOTE)
            # ======================
            with st.expander("🔍 View Resampled Dataset"):
                preview_resampled = pd.DataFrame(X_resampled, columns=X.columns)
                preview_resampled[target] = y_resampled
                st.dataframe(preview_resampled.head(100), width='stretch')

            # ======================
            # SAVE RESULT (SESSION)
            # ======================
            st.session_state["X_model"] = X_resampled
            st.session_state["y_model"] = y_resampled
            st.session_state["use_smote"] = True

            st.session_state["model_data_info"] = {
                "source": "smote",
                "rows": len(y_resampled),
                "features": X_resampled.shape[1]
            }

            # 🔥 SAVE TO DB (METADATA)
            save_model_dataset(
                project_id,
                df_woe,
                selected_features,
                source="smote"
            )

            st.success("Resampled dataset ready for modelling!")

        except Exception as e:
            st.error(f"SMOTE failed: {e}")

    # ======================
    # SHOW CURRENT DATA SOURCE
    # ======================
    if "model_data_info" in st.session_state:

        info = st.session_state["model_data_info"]

        st.subheader("📌 Current Modelling Dataset")

        if info["source"] == "smote":
            st.success("Using SMOTE dataset")
        else:
            st.info("Using original dataset")

        st.write(f"Rows: {info['rows']}")
        st.write(f"Features: {info['features']}")
