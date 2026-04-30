import streamlit as st
import pandas as pd
import numpy as np

from database.crud import load_split, load_preprocessing, save_binning, load_binning
from utils.binning import (
    create_numeric_bins,
    create_categorical_bins,
    calculate_bin_stats,
    create_manual_categorical_bins,
    create_manual_numeric_bins,
    create_optimal_bins
)

# ======================
# HELPER: SORT NUMERIC BIN
# ======================
def get_lower(x):
    try:
        s = str(x)
        lower = s.split(",")[0]
        lower = lower.replace("(", "").replace("[", "").strip()

        if lower == "-inf":
            return float("-inf")
        if lower == "inf":
            return float("inf")

        return float(lower)

    except:
        return float("-inf")


def run(project_id):
    st.header("📊 Binning")

    # ======================
    # LOAD DATA
    # ======================
    split = load_split(project_id)
    config = load_preprocessing(project_id)
    saved_rules = load_binning(project_id)  # 🔥 NEW

    if saved_rules is None:
        saved_rules = {}

    if split is None or config is None:
        st.warning("Complete previous steps first")
        return

    df = split["train"].copy()
    target = config["target"]
    features = config["features"]

    # ======================
    # VARIABLE SELECTION (NEW)
    # ======================
    st.subheader("📌 Select Variables for Binning")

    selected_features = st.multiselect(
        "Choose variables",
        features,
        default=features
    )    

    st.success(f"Using TRAIN data: {df.shape}")

    binning_rules = {}

    # ======================
    # LOOP FEATURES
    # ======================
    for col in selected_features:
        st.subheader(f"🔹 {col}")

        col_data_numeric = pd.to_numeric(df[col], errors='coerce')
        is_numeric = col_data_numeric.notna().sum() > 0.8 * len(df)

        # ======================
        # 🔥 LOAD TRANSFORM DEFAULT
        # ======================
        transform = {"type": "none"}
        default_log = False

        if col in saved_rules:
            t = saved_rules[col].get("transform", {})
            if t.get("type") == "log1p":
                default_log = True

        # ======================
        # TRANSFORMATION
        # ======================
        if is_numeric:
            st.write("### 🔄 Transformation")

            use_log = st.checkbox(
                f"Apply log1p transformation for {col}",
                value=default_log,
                key=f"{col}_log"
            )

            if use_log:
                min_val = col_data_numeric.min()

                if min_val <= -1:
                    shift = abs(min_val) + 1
                    st.caption(f"Auto shift applied: +{shift:.4f}")
                    col_data_numeric = np.log1p(col_data_numeric + shift)
                    transform = {"type": "log1p", "shift": float(shift)}
                else:
                    col_data_numeric = np.log1p(col_data_numeric)
                    transform = {"type": "log1p", "shift": 0.0}

                st.success("Log1p transformation applied")

        # ======================
        # LOAD MODE DEFAULT
        # ======================
        default_mode = "Quantile"

        if col in saved_rules:
            m = saved_rules[col].get("mode")
            if m == "quantile":
                default_mode = "Quantile"
            elif m == "optimal":
                default_mode = "Optimal (optbinning)"
            elif m == "manual":
                default_mode = "Manual"

        mode = st.radio(
            f"Binning Mode for {col}",
            ["Quantile", "Optimal (optbinning)", "Manual"],
            index=["Quantile", "Optimal (optbinning)", "Manual"].index(default_mode),
            key=f"{col}_mode"
        )

        # ======================
        # QUANTILE
        # ======================
        if mode == "Quantile":

            if is_numeric:
                default_bins = 5
                if col in saved_rules:
                    default_bins = saved_rules[col].get("n_bins", 5)

                n_bins = st.slider(
                    f"Number of bins",
                    2, 10, default_bins,
                    key=f"{col}_bins"
                )

                bins = create_numeric_bins(col_data_numeric, n_bins)

            else:
                bins = create_categorical_bins(df[col])

        # ======================
        # OPTIMAL BINNING
        # ======================
        elif mode == "Optimal (optbinning)":

            if not is_numeric:
                st.warning("Optbinning only supports numeric variables")
                continue

            st.write("### ⚙️ Optbinning Settings")

            monotonic = st.selectbox(
                f"Monotonic Trend for {col}",
                ["auto", "ascending", "descending"],
                key=f"{col}_mono"
            )

            sample_size = min(len(df), 5000)
            df_sample = df.sample(sample_size, random_state=42)

            with st.spinner(f"Running optimal binning for {col}..."):
                try:
                    bins, optb_model = create_optimal_bins(
                        df_sample[col],
                        df_sample[target],
                        monotonic_trend=monotonic
                    )

                    st.success("Optimal binning created")
                    st.caption(f"Using sample size: {sample_size}")

                    st.write("### Bin Splits")
                    st.write(optb_model.splits)

                except Exception as e:
                    st.error(f"Optbinning failed: {e}")
                    st.warning("Fallback to quantile binning")
                    bins = create_numeric_bins(col_data_numeric, 5)

        # ======================
        # MANUAL BINNING
        # ======================
        else:

            if is_numeric:

                col_data = col_data_numeric

                st.write("### 📊 Data Insight")

                st.write({
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "median": float(col_data.median())
                })

                quantiles = col_data.quantile([0.2, 0.4, 0.6, 0.8]).values
                suggested = [round(q, 2) for q in quantiles]

                st.write("Suggested cut points:")
                st.code(", ".join(map(str, suggested)))

                if st.button(f"Use Suggested Cuts for {col}"):
                    st.session_state[f"{col}_cut"] = ", ".join(map(str, suggested))

                default_cut = ""
                if col in saved_rules and "cut_points" in saved_rules[col]:
                    default_cut = ", ".join(map(str, saved_rules[col]["cut_points"]))

                input_cut = st.text_input(
                    f"Cut points for {col}",
                    value=default_cut,
                    key=f"{col}_cut"
                )

                if input_cut:
                    try:
                        cut_points = [float(x.strip()) for x in input_cut.split(",")]
                        bins = create_manual_numeric_bins(col_data, cut_points)
                    except:
                        st.error("Invalid input")
                        continue
                else:
                    st.info("Please input cut points")
                    continue

            else:
                st.write("### Value Distribution")

                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = ["Value", "Count"]

                threshold = 0.05 * len(df)

                value_counts["Group"] = value_counts.apply(
                    lambda x: "Other" if x["Count"] < threshold else x["Value"],
                    axis=1
                )

                # 🔥 LOAD MAPPING DEFAULT
                if col in saved_rules and "mapping" in saved_rules[col]:
                    mapping_saved = saved_rules[col]["mapping"]
                    value_counts["Group"] = value_counts["Value"].map(mapping_saved).fillna(value_counts["Value"])

                edited_df = st.data_editor(
                    value_counts,
                    width='stretch',
                    hide_index=True,
                    key=f"{col}_group_editor"
                )

                mapping = dict(zip(edited_df["Value"], edited_df["Group"]))
                bins = create_manual_categorical_bins(df[col], mapping)

        # ======================
        # FINAL RESULT
        # ======================
        try:
            result = calculate_bin_stats(df, col, target, bins)

            if is_numeric:
                result["lower_bound"] = result["feature"].apply(get_lower)
                result = result.sort_values(by="lower_bound").reset_index(drop=True)
                result = result.drop(columns=["lower_bound"])

            else:
                result = result.sort_values(by="bad_ratio", ascending=True).reset_index(drop=True)
                result["risk_rank"] = range(1, len(result) + 1)

            st.write("### Final Binning Result")
            st.dataframe(result, width='stretch')

        except Exception as e:
            st.error(f"Error in binning: {e}")
            continue

        # ======================
        # AUTO REFERENCE
        # ======================
        if is_numeric:
            try:
                auto_bins = create_numeric_bins(col_data_numeric, 5)
                auto_result = calculate_bin_stats(df, col, target, auto_bins)

                st.write("### Auto Binning Reference")
                st.dataframe(auto_result, width='stretch')

            except:
                st.warning("Auto binning reference failed")

        # ======================
        # SAVE RULE
        # ======================
        if is_numeric:

            if mode == "Quantile":
                binning_rules[col] = {
                    "type": "numeric",
                    "mode": "quantile",
                    "n_bins": n_bins,
                    "transform": transform
                }

            elif mode == "Optimal (optbinning)":
                binning_rules[col] = {
                    "type": "numeric",
                    "mode": "optimal",
                    "splits": optb_model.splits.tolist(),
                    "transform": transform
                }

            else:
                binning_rules[col] = {
                    "type": "numeric",
                    "mode": "manual",
                    "cut_points": cut_points,
                    "transform": transform
                }

        else:
            if mode == "Quantile":
                binning_rules[col] = {
                    "type": "categorical",
                    "mode": "quantile"
                }
            else:
                binning_rules[col] = {
                    "type": "categorical",
                    "mode": "manual",
                    "mapping": mapping
                }

    # ======================
    # SAVE
    # ======================
    if st.button("💾 Save Binning"):
        save_binning(project_id, binning_rules)
        st.success("Binning rules saved!")
