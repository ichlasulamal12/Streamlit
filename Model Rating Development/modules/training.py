import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import ast

import statsmodels.api as sm

from database.crud import (
    load_split,
    load_preprocessing,
    load_model_dataset,
    load_binning,
    save_model_dataset
)
from database.db import get_connection


# ======================
# SAFE PARSER
# ======================
def parse_features(raw):
    if raw is None:
        return None

    if isinstance(raw, list):
        return raw

    try:
        return json.loads(raw)
    except:
        return ast.literal_eval(raw)


# ======================
# SAVE MODEL
# ======================
def save_model(project_id, model, features):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_result (
            project_id INTEGER PRIMARY KEY,
            model BLOB,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT OR REPLACE INTO model_result (project_id, model, features)
        VALUES (?, ?, ?)
    """, (
        project_id,
        pickle.dumps(model),
        json.dumps(features)
    ))

    conn.commit()
    conn.close()


# ======================
# SAVE CALIBRATED MODEL
# ======================
def save_calibrated_model(project_id, params, features):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_calibrated (
            project_id INTEGER PRIMARY KEY,
            params BLOB,
            features TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT OR REPLACE INTO model_calibrated (project_id, params, features)
        VALUES (?, ?, ?)
    """, (
        project_id,
        pickle.dumps(params),
        json.dumps(features)
    ))

    conn.commit()
    conn.close()


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


# ======================
# DETECT TREND (FIXED: USE WOE RESULT)
# ======================
def detect_trend_from_woe(woe_result, var):

    df = woe_result[woe_result["variabel"] == var].copy()

    if df.empty:
        return "unknown", df

    try:
        df = df.sort_values(by="kategori")
    except:
        df = df.sort_values(by="woe")

    woe_values = df["woe"].values

    if len(woe_values) < 3:
        return "unknown", df

    diff = np.diff(woe_values)

    if np.all(diff >= 0):
        return "ascending", df
    elif np.all(diff <= 0):
        return "descending", df
    else:
        return "non_monotonic", df


# ======================
# MAIN
# ======================
def run(project_id):

    st.header("📊 Model Training (Logistic Regression)")

    split = load_split(project_id)
    config = load_preprocessing(project_id)
    model_data = load_model_dataset(project_id)

    if split is None or config is None or model_data is None:
        st.warning("Complete previous steps first")
        return

    target = config["target"]

    df_train = split["train"].copy()
    y = df_train[target]

    df_woe = model_data["df_woe"]
    features = model_data["features"]
    woe_result = model_data.get("woe_result")

    if woe_result is None:
        st.error("WOE not found. Run WOE module first.")
        return

    # ======================
    # LOAD EXISTING MODEL
    # ======================
    saved_model, saved_features = load_model(project_id)

    if saved_model is not None:
        st.success("Loaded saved model from database")
        st.session_state["model"] = saved_model
        st.session_state["model_features"] = saved_features

    # ======================
    # USE DATA
    # ======================
    if "X_model" in st.session_state and "y_model" in st.session_state:
        X = st.session_state["X_model"]
        y = st.session_state["y_model"]
        st.success("Using dataset from SMOTE / latest step")
    else:
        X = df_woe[features]
        st.info("Using original WOE dataset")

    # ======================
    # FEATURE SELECTION
    # ======================
    st.subheader("✏️ Select Variables")

    selected_vars = st.multiselect(
        "Variables for training",
        options=X.columns.tolist(),
        default=X.columns.tolist()
    )

    if not selected_vars:
        st.warning("Select at least one variable")
        return

    X = X[selected_vars]
    X_const = sm.add_constant(X)

    # ======================
    # TRAIN
    # ======================
    if st.button("🚀 Train Model"):

        try:
            model = sm.Logit(y, X_const).fit(disp=0)

            # ======================
            # BUILD SCORECARD
            # ======================
            coef_df = pd.DataFrame({
                "index": model.params.index,
                "Coefficient_final": model.params.values,
                "std_error": model.bse.values,
                "p_value": model.pvalues.values
            })

            intercept = model.params.get("const", 0)
            woe_result["kategori"] = woe_result["kategori"].astype(str)

            # ======================
            # SAVE MODEL DATASET
            # ======================
            save_model_dataset(
                project_id=project_id,
                df_woe=df_woe,
                features=selected_vars,
                woe_result=woe_result,  # ✅ dari modul WOE
                coef_df=coef_df,
                intercept=intercept
            )

            st.success("Model trained successfully")

            # ======================
            # DISPLAY
            # ======================
            st.subheader("📊 Model Coefficients")
            st.dataframe(coef_df)

            st.subheader("📄 Model Summary")
            st.text(model.summary())

            st.subheader("⭐ Feature Importance")

            importance = pd.DataFrame({
                "variable": model.params.index,
                "importance": np.abs(model.params.values)
            }).sort_values(by="importance", ascending=False)

            st.dataframe(importance)

            # ======================
            # SIGN CHECK (FIXED)
            # ======================
            st.subheader("🔍 Sign Check (WOE vs Coefficient)")

            warnings = []

            for var in selected_vars:

                if var not in woe_result["variabel"].unique():
                    continue

                coef = model.params.get(var)

                if coef is None:
                    continue

                trend, grouped = detect_trend_from_woe(woe_result, var)

                sample_value = grouped["kategori"].iloc[0]

                is_categorical = not (
                    pd.api.types.is_numeric_dtype(grouped["kategori"])
                    or ("(" in str(sample_value) and "," in str(sample_value))
                )

                if is_categorical:
                    st.info(f"{var}: ℹ️ Categorical variable (sign check skipped)")
                    continue

                if trend == "ascending" and coef < 0:
                    warnings.append(f"{var}: ❌ WOE naik tapi coef negatif")

                elif trend == "descending" and coef > 0:
                    warnings.append(f"{var}: ❌ WOE turun tapi coef positif")

                elif trend == "non_monotonic":
                    st.warning(f"{var}: ⚠️ Non-monotonic WOE")

                with st.expander(f"📊 {var} WOE Trend"):
                    st.dataframe(grouped)

                    if "woe" in grouped.columns:
                        st.line_chart(grouped["woe"])

            if warnings:
                for w in warnings:
                    st.error(w)
            else:
                st.success("All variable signs are consistent with WOE")

            st.subheader("📈 Model Performance")
            st.write(f"Pseudo R² (McFadden): {model.prsquared:.4f}")

            # ======================
            # SAVE MODEL
            # ======================
            save_model(project_id, model, selected_vars)

            st.session_state["model"] = model
            st.session_state["model_features"] = selected_vars

        except Exception as e:
            st.error(f"Model training failed: {e}")

    # ======================
    # CALIBRATION
    # ======================
    if "model" in st.session_state:

        st.subheader("🎯 Intercept Calibration")

        model = st.session_state["model"]
        selected_vars = st.session_state["model_features"]

        X_current = X[selected_vars]
        X_const_current = sm.add_constant(X_current)

        actual_pd = df_train[target].mean()
        pred_pd = model.predict(X_const_current).mean()

        st.write(f"Actual PD: {actual_pd:.4f}")
        st.write(f"Model PD: {pred_pd:.4f}")

        if st.button("⚖️ Calibrate Intercept"):

            actual_odds = actual_pd / (1 - actual_pd)
            pred_odds = pred_pd / (1 - pred_pd)

            adjustment = np.log(actual_odds / pred_odds)

            adjusted_params = model.params.copy()
            adjusted_params["const"] += adjustment

            st.subheader("📊 Calibrated Coefficients")

            st.dataframe(pd.DataFrame({
                "variable": adjusted_params.index,
                "coefficient": adjusted_params.values
            }))

            save_calibrated_model(
                project_id,
                adjusted_params,
                adjusted_params.index.tolist()
            )

            st.success("Calibrated model saved")
