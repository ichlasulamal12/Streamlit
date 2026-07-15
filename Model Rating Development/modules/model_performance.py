import streamlit as st
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pickle
import json
import ast
import matplotlib.pyplot as plt

from utils.binning import apply_binning
from utils.transform import apply_transformation

from database.crud import (
    load_split,
    load_preprocessing,
    load_model_dataset,
    load_binning,
    load_model_rules, 
    save_model_rules
)
from database.db import get_connection


# ======================
# SAFE PARSER
# ======================
def parse_features(raw):
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except:
        return ast.literal_eval(raw)


# ======================
# ALIGN FEATURES
# ======================
def align_features(X, feature_list):
    X = X.copy()

    for col in feature_list:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_list]
    return X


def prepare_exog(X, feature_list):
    if "const" in feature_list:
        cols = [c for c in feature_list if c != "const"]
        X = align_features(X, cols)
        X = sm.add_constant(X, has_constant='add')
    else:
        X = align_features(X, feature_list)

    X = X[feature_list]
    return X


# ======================
# APPLY WOE
# ======================
def apply_woe_from_result(df, woe_result):
    df = df.copy()

    for var in woe_result['variabel'].unique():
        mapping = (
            woe_result[woe_result["variabel"] == var]
            .set_index("kategori")["woe"]
        )

        missing_woe = mapping.get("Missing", 0)

        df[var] = df[var].map(mapping)
        df[var] = df[var].fillna(missing_woe)

    return df


# ======================
# LOAD MODEL
# ======================
def load_model(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT model, features
        FROM model_result
        WHERE project_id = ?
    """, (project_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None, None

    return pickle.loads(row[0]), parse_features(row[1])


def load_calibrated_model(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT params, features
        FROM model_calibrated
        WHERE project_id = ?
    """, (project_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None, None

    return pickle.loads(row[0]), parse_features(row[1])


# ======================
# METRICS
# ======================
def build_performance_table_rating(df_perf, group_col, rules):

    df = df_perf.copy()

    # ======================
    # AGGREGATION
    # ======================
    grouped = df.groupby(group_col)["target"].agg(
        total="count",
        bad="sum"
    ).reset_index()

    grouped["good"] = grouped["total"] - grouped["bad"]

    # ======================
    # ADD BOUNDARIES
    # ======================
    rule_df = pd.DataFrame(rules, columns=["name", "min", "max"])

    grouped = grouped.merge(
        rule_df,
        left_on=group_col,
        right_on="name",
        how="left"
    )

    # ======================
    # SORT (IMPORTANT)
    # ======================
    grouped = grouped.sort_values("min", ascending=True).reset_index(drop=True)

    # ======================
    # TOTALS
    # ======================
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()
    total_all = grouped["total"].sum()

    # ======================
    # RATIOS
    # ======================
    grouped["bad_ratio"] = grouped["bad"] / grouped["total"]

    grouped["good_pct"] = grouped["good"] / total_good
    grouped["bad_pct"] = grouped["bad"] / total_bad
    grouped["total_pct"] = grouped["total"] / total_all

    # ======================
    # CUMULATIVE (BOTTOM → UP)
    # ======================
    grouped["cum_good_pct"] = grouped["good_pct"][::-1].cumsum()[::-1]
    grouped["cum_bad_pct"] = grouped["bad_pct"][::-1].cumsum()[::-1]
    grouped["cum_total_pct"] = grouped["total_pct"][::-1].cumsum()[::-1]

    # ======================
    # KS (shift n+1)
    # ======================
    grouped["cum_good_next"] = grouped["cum_good_pct"].shift(-1)
    grouped["cum_bad_next"] = grouped["cum_bad_pct"].shift(-1)

    grouped["KS"] = (grouped["cum_good_next"] - grouped["cum_bad_next"]).abs()

    # ======================
    # ROC (your formula)
    # ======================
    grouped["ROC"] = (
        0.5 * grouped["good_pct"] * grouped["bad_pct"]
        + (1 - grouped["cum_good_pct"]) * grouped["bad_pct"]
    )

    # ======================
    # CLEAN COLUMNS
    # ======================
    grouped = grouped.rename(columns={
        group_col: "bucket",
        "min": "min_bound",
        "max": "max_bound"
    })

    grouped = grouped[[
        "bucket",
        "min_bound",
        "max_bound",
        "good",
        "bad",
        "total",
        "bad_ratio",
        "good_pct",
        "bad_pct",
        "total_pct",
        "cum_good_pct",
        "cum_bad_pct",
        "cum_total_pct",
        "KS",
        "ROC"
    ]]

    return grouped

def build_performance_table_score(df_perf, group_col, rules):

    df = df_perf.copy()

    # ======================
    # AGGREGATION
    # ======================
    grouped = df.groupby(group_col)["target"].agg(
        total="count",
        bad="sum"
    ).reset_index()

    grouped["good"] = grouped["total"] - grouped["bad"]

    # ======================
    # ADD BOUNDARIES
    # ======================
    rule_df = pd.DataFrame(rules, columns=["name", "min", "max"])

    grouped = grouped.merge(
        rule_df,
        left_on=group_col,
        right_on="name",
        how="left"
    )

    # ======================
    # SORT (IMPORTANT)
    # ======================
    grouped = grouped.sort_values("min", ascending=False).reset_index(drop=True)

    # ======================
    # TOTALS
    # ======================
    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()
    total_all = grouped["total"].sum()

    # ======================
    # RATIOS
    # ======================
    grouped["bad_ratio"] = grouped["bad"] / grouped["total"]

    grouped["good_pct"] = grouped["good"] / total_good
    grouped["bad_pct"] = grouped["bad"] / total_bad
    grouped["total_pct"] = grouped["total"] / total_all

    # ======================
    # CUMULATIVE (BOTTOM → UP)
    # ======================
    grouped["cum_good_pct"] = grouped["good_pct"][::-1].cumsum()[::-1]
    grouped["cum_bad_pct"] = grouped["bad_pct"][::-1].cumsum()[::-1]
    grouped["cum_total_pct"] = grouped["total_pct"][::-1].cumsum()[::-1]

    # ======================
    # KS (shift n+1)
    # ======================
    grouped["cum_good_next"] = grouped["cum_good_pct"].shift(-1)
    grouped["cum_bad_next"] = grouped["cum_bad_pct"].shift(-1)

    grouped["KS"] = (grouped["cum_good_next"] - grouped["cum_bad_next"]).abs()

    # ======================
    # ROC (your formula)
    # ======================
    grouped["ROC"] = (
        0.5 * grouped["good_pct"] * grouped["bad_pct"]
        + (1 - grouped["cum_good_pct"]) * grouped["bad_pct"]
    )

    # ======================
    # CLEAN COLUMNS
    # ======================
    grouped = grouped.rename(columns={
        group_col: "bucket",
        "min": "min_bound",
        "max": "max_bound"
    })

    grouped = grouped[[
        "bucket",
        "min_bound",
        "max_bound",
        "good",
        "bad",
        "total",
        "bad_ratio",
        "good_pct",
        "bad_pct",
        "total_pct",
        "cum_good_pct",
        "cum_bad_pct",
        "cum_total_pct",
        "KS",
        "ROC"
    ]]

    return grouped

# ======================
# SCORE FUNCTION
# ======================
def get_score(df, df_scorecard, intercept_score):
    df_score = df.copy()

    for var in df_scorecard['variabel'].unique():
        score_map = df_scorecard[
            df_scorecard['variabel'] == var
        ].set_index('kategori')['Score']

        missing_score = score_map.get("Missing", 0)

        df_score[var + "_Score"] = (
            df_score[var]
            .map(score_map)
            .fillna(missing_score)
        )

    score_cols = [c for c in df_score.columns if c.endswith('_Score')]

    df_score["Score_total"] = (
        df_score[score_cols].sum(axis=1)
        + intercept_score
    )

    return df_score


# ======================
# HELPER: MANUAL MAPPING (IMPORTANT)
# ======================
def map_with_rules(x, rules):
    for i, (name, low, high) in enumerate(rules):
        if i == len(rules) - 1:
            if low <= x <= high:
                return name
        else:
            if low <= x < high:
                return name
    return "Unknown"


# ======================
# MAIN
# ======================
def run(project_id):

    # ======================
    # SYNC WITH SAVED RULES
    # ======================
    if "n_rating" not in st.session_state:
        if "rating_rules" in st.session_state and len(st.session_state["rating_rules"]) > 0:
            st.session_state["n_rating"] = len(st.session_state["rating_rules"])
        else:
            st.session_state["n_rating"] = 7

    if "n_score_bins" not in st.session_state:
        if "score_rules" in st.session_state and len(st.session_state["score_rules"]) > 0:
            st.session_state["n_score_bins"] = len(st.session_state["score_rules"])
        else:
            st.session_state["n_score_bins"] = 7

    if "rating_rules" not in st.session_state or "score_rules" not in st.session_state:
        rating_db, score_db = load_model_rules(project_id)

        st.session_state["rating_rules"] = rating_db if rating_db else []
        st.session_state["score_rules"] = score_db if score_db else []
        
    st.header("📊 Model Performance")

    split = load_split(project_id)
    config = load_preprocessing(project_id)
    model_data = load_model_dataset(project_id)
    binning_rules = load_binning(project_id)

    if split is None or config is None or model_data is None:
        st.warning("Complete previous steps first")
        return

    target = config["target"]

    df_train = split["train"]
    df_test = split["test"]
    df_val = split.get("val")

    features = model_data["features"]

    model, _ = load_model(project_id)
    calibrated_params, calibrated_features = load_calibrated_model(project_id)

    if model is None:
        st.warning("Train model first")
        return

    # ======================
    # UI
    # ======================
    st.subheader("⚙️ Setup")

    col1, col2, col3 = st.columns(3)

    with col1:
        model_type = st.radio("Model", ["Original", "Calibrated"])

    with col2:
        dataset_type = st.radio("Dataset", ["Train", "Test", "Validation"])

    with col3:
        output_type = st.radio("Output", ["Rating", "Score"])

    # ======================
    # DATA
    # ======================
    if dataset_type == "Train":
        df_raw = df_train.copy()
        y_true = df_train[target]
    elif dataset_type == "Test":
        df_raw = df_test.copy()
        y_true = df_test[target]
    else:
        if df_val is None:
            st.warning("No validation data available")
            return
        df_raw = df_val.copy()
        y_true = df_val[target]

    df_binned = apply_binning(
        apply_transformation(df_raw.copy(), binning_rules),
        binning_rules
    )

    woe_result = model_data["woe_result"]
    woe_result["kategori"] = woe_result["kategori"].astype(str)
    df_binned = df_binned.astype(str)

    df_model = apply_woe_from_result(df_binned, woe_result)
    # 🔥 pastikan tidak ada NaN
    df_model = df_model.fillna(0)
    X = df_model[features].copy()
    X = X.fillna(0)

    # ======================
    # PREDICT
    # ======================
    if model_type == "Original":
        cols = model.model.exog_names
        X_model = prepare_exog(X, cols)
        y_prob = model.predict(X_model)
    else:
        cols = calibrated_features
        X_model = prepare_exog(X, cols)
        linear = np.dot(X_model.values, calibrated_params)
        y_prob = 1 / (1 + np.exp(-linear))

    # ======================
    # RATING
    # ======================
    if output_type == "Rating":

        st.subheader("🏷️ Rating Setup")

        n = st.number_input(
            "Number of Rating",
            min_value=2,
            max_value=10,
            value=st.session_state["n_rating"],
            key="n_rating"
        )

        rating_rules = []

        for i in range(n):

            # ambil default dari session jika ada
            if i < len(st.session_state["rating_rules"]):
                default_name, default_low, default_high = st.session_state["rating_rules"][i]
            else:
                default_name, default_low, default_high = f"Grade {i+1}", 0.0, 1.0

            c1, c2, c3 = st.columns(3)

            name = c1.text_input(f"Rating {i+1}", value=default_name, key=f"r{i}")

            low = c2.number_input(
                f"Min {i+1}",
                value=float(default_low),
                format="%.10f",
                step=0.0001,
                key=f"l{i}"
            )

            high = c3.number_input(
                f"Max {i+1}",
                value=float(default_high),
                format="%.10f",
                step=0.0001,
                key=f"h{i}"
            )

            rating_rules.append((name, low, high))

        # ======================
        # RESET BUTTON (DI SINI)
        # ======================
        col_reset, col_space = st.columns([1,5])

        with col_reset:
            if st.button("🔄 Reset Rating"):
                st.session_state["rating_rules"] = []

                save_model_rules(
                    project_id,
                    rating_rules=[],
                    score_rules=st.session_state.get("score_rules")
                )

                st.rerun()

        # simpan kembali
        st.session_state["rating_rules"] = rating_rules

        save_model_rules(
            project_id,
            rating_rules=rating_rules,
            score_rules=st.session_state.get("score_rules")
        )

        df_perf = pd.DataFrame({
            "prob": y_prob,
            "target": y_true
        }).reset_index(drop=True)

        df_perf = df_perf.sort_values("prob", ascending=True).reset_index(drop=True)

        df_perf["freq_cum"] = np.arange(1, len(df_perf)+1)
        df_perf["freq_bad_cum"] = df_perf["target"].cumsum()

        total = len(df_perf)
        total_bad = df_perf["target"].sum()

        df_perf["cum_portion"] = df_perf["freq_cum"] / total
        df_perf["cum_bad"] = df_perf["freq_bad_cum"] / total_bad

        df_perf["rating"] = df_perf["prob"].apply(lambda x: map_with_rules(x, rating_rules))

        st.dataframe(df_perf.style.format({"prob": "{:.10f}"}))

    # ======================
    # SCORE
    # ======================
    else:

        st.subheader("📋 Scorecard Table (Per Bin)")

        woe_result = model_data["woe_result"]
        coef_df = model_data["coef_df"]
        intercept = model_data["intercept"]

        BaseScore = 600
        ReferenceScore = 600
        PDO = 50
        ReferenceOdds = 50

        Factor = PDO / np.log(2)

        Offset = (
            ReferenceScore
            - Factor * np.log(ReferenceOdds)
        )

        InterceptScore = (
            Offset
            - Factor * intercept
        )        

        st.info(f"""
        ### Scorecard Configuration

        - Base Score      : {BaseScore}
        - PDO             : {PDO}
        - Reference Odds  : {ReferenceOdds}:1
        - Factor          : {Factor:.6f}
        - Offset          : {Offset:.4f}
        - Intercept Score : {InterceptScore:.4f}
        """)

        df_score = woe_result.merge(
            coef_df,
            left_on='variabel',
            right_on='index'
        )

        # 🔥 BIN-LEVEL SCALING
        df_score['Score'] = - Factor * df_score['Coefficient_final'] * df_score['woe']
        df_score = df_score.dropna()

        st.dataframe(
            df_score[['variabel','kategori','woe','Coefficient_final','Score']]
            .sort_values(['variabel','kategori']),
            width='stretch'
        )

        st.write("Intercept Score:", round(Offset, 2))

        df_score_result = get_score(df_binned, df_score, InterceptScore)

        df_score_result["Odds"] = np.exp(
            (df_score_result["Score_total"] - Offset) / Factor
        )

        df_score_result["PD_from_Score"] = (
            1 / (1 + df_score_result["Odds"])
        )        

        st.subheader("📊 Score Result")

        st.dataframe(
            df_score_result[
                [
                    "Score_total",
                    "Odds",
                    "PD_from_Score"
                ]
            ].style.format({
                "Score_total": "{:.0f}",
                "Odds": "{:.2f}",
                "PD_from_Score": "{:.4%}"
            }),
            width="stretch"
        )

        # ===== RANGE SETUP =====
        st.subheader("🎯 Score Range Setup")

        n_bins = st.number_input(
            "Number of Score Bins",
            min_value=2,
            max_value=10,
            value=st.session_state["n_score_bins"],
            key="n_score_bins"
        )

        score_rules = []

        for i in range(n_bins):

            if i < len(st.session_state["score_rules"]):
                default_name, default_low, default_high = st.session_state["score_rules"][i]
            else:
                default_name, default_low, default_high = f"Range {i+1}", 300, 850

            c1, c2, c3 = st.columns(3)

            name = c1.text_input(f"Range {i+1}", value=default_name, key=f"s{i}")

            low = c2.number_input(
                f"Min Score {i+1}",
                value=int(default_low),
                format="%d",
                step=1,
                key=f"sl{i}"
            )

            high = c3.number_input(
                f"Max Score {i+1}",
                value=int(default_high),
                format="%d",
                step=1,
                key=f"sh{i}"
            )

            score_rules.append((name, low, high))

        # ======================
        # RESET BUTTON
        # ======================
        col_reset, col_space = st.columns([1,5])

        with col_reset:
            if st.button("🔄 Reset Score"):
                st.session_state["score_rules"] = []

                save_model_rules(
                    project_id,
                    rating_rules=st.session_state.get("rating_rules"),
                    score_rules=[]
                )

                st.rerun()

        st.session_state["score_rules"] = score_rules

        save_model_rules(
            project_id,
            rating_rules=st.session_state.get("rating_rules"),
            score_rules=score_rules
        )

        # ===== PERFORMANCE TABLE =====
        df_perf = pd.DataFrame({
            "score": df_score_result["Score_total"],
            "target": y_true
        }).reset_index(drop=True)

        df_perf = df_perf.sort_values("score", ascending=False).reset_index(drop=True)

        df_perf["freq_cum"] = np.arange(1, len(df_perf)+1)
        df_perf["freq_bad_cum"] = df_perf["target"].cumsum()

        total = len(df_perf)
        total_bad = df_perf["target"].sum()

        df_perf["cum_portion"] = df_perf["freq_cum"] / total
        df_perf["cum_bad"] = df_perf["freq_bad_cum"] / total_bad

        df_perf["score_range"] = df_perf["score"].apply(lambda x: map_with_rules(x, score_rules))

        st.dataframe(df_perf.style.format({"prob": "{:.10f}"}))

    # ======================
    # PERFORMANCE
    # ======================
    if st.button("📈 Calculate Performance"):

        if output_type == "Rating":

            perf_table = build_performance_table_rating(
                df_perf,
                "rating",
                rating_rules
            )

        else:

            perf_table = build_performance_table_score(
                df_perf,
                "score_range",
                score_rules
            )

        # ======================
        # METRICS
        # ======================
        ks_value = perf_table["KS"].max()
        auroc = perf_table["ROC"].sum()
        gini = 2 * auroc - 1

        # ======================
        # DISPLAY
        # ======================
        st.subheader("📊 Performance Table")

        st.dataframe(
            perf_table.style.format({
                "bad_ratio": "{:.4f}",
                "good_pct": "{:.4f}",
                "bad_pct": "{:.4f}",
                "total_pct": "{:.4f}",
                "cum_good_pct": "{:.4f}",
                "cum_bad_pct": "{:.4f}",
                "cum_total_pct": "{:.4f}",
                "KS": "{:.4f}",
                "ROC": "{:.4f}"
            }),
            width="stretch"
        )

        st.subheader("📈 Metrics")

        st.write(f"KS: {ks_value:.4f}")
        st.write(f"AUROC: {auroc:.4f}")
        st.write(f"Gini: {gini:.4f}")

        st.subheader("📈 ROC Curve")

        # ======================
        # PREPARE DATA PLOT
        # ======================
        roc_df = perf_table.copy()

        # balik urutan: dari BEST → WORST
        roc_df = roc_df.iloc[::-1].reset_index(drop=True)

        # ======================
        # TAMBAH (0,0)
        # ======================
        roc_x = np.insert(roc_df["cum_good_pct"].values, 0, 0)
        roc_y = np.insert(roc_df["cum_bad_pct"].values, 0, 0)

        # ======================
        # PLOT
        # ======================
        fig, ax = plt.subplots()

        ax.plot(roc_x, roc_y, marker='o')
        ax.plot([0, 1], [0, 1], linestyle='--')  # diagonal reference
        ax.fill_between(roc_x, roc_y, alpha=0.2)
        ax.set_xlabel("Cumulative Good %")
        ax.set_ylabel("Cumulative Bad %")
        ax.set_title("ROC Curve")

        ax.grid(True)

        st.pyplot(fig)        
