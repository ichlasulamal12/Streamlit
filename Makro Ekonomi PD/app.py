# =========================================================
# CREDIT RISK MODELLING APPLICATION
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    linear_reset
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import shapiro
from pmdarima import auto_arima

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table
from reportlab.lib.styles import getSampleStyleSheet
import io

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(layout="wide")

# =========================================================
# PROFESSIONAL FONT (IBM PLEX SANS)
# =========================================================

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"]  {
    font-family: 'IBM Plex Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CORPORATE CLEAN STYLE
# =========================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #F4F6F9;
}

/* Headings */
h1 {
    color: #1F2A44;
    font-weight: 600;
}

h2, h3 {
    color: #243B6B;
    font-weight: 500;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1F2937;
}

section[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

/* Metric Card */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
}

/* Expander */
details {
    background-color: #FFFFFF;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
    padding: 8px;
}

/* Dataframe */
.stDataFrame {
    background-color: #FFFFFF;
}

/* Buttons */
button[kind="primary"] {
    background-color: #1F4E79 !important;
    border-radius: 8px !important;
    color: white !important;
}

button[kind="secondary"] {
    border-radius: 8px !important;
}

/* Divider spacing */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

st.title("Credit Risk Modelling Application")

st.markdown("""
**Integrated Macro Forecasting & PD MEV Forecast Engine**  
Internal Quantitative Risk Tool
""")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.header("📂 Navigation")

    active_tab = st.radio(
        "Select Module",
        ["📈 Macro Forecast", "📊 PD MEV Forecast"],
        key="active_module"
    )
    st.divider()

tab1, tab2 = st.tabs([
    "📈 Macro Forecast",
    "📊 PD MEV Forecast"
])

# =========================================================
# FORMAT HELPER
# =========================================================

def fmt4(x):
    return round(float(x), 4)

def fmt6(x):
    return round(float(x), 6)

def fmt_pct(x):
    return f"{x*100:.2f}%"

# =========================================================
# ========================= TAB 1 =========================
# =========================================================

with tab1:

    if active_tab == "📈 Macro Forecast":

        with st.sidebar:
            st.subheader("📈 Macro Forecast Controls")

            file = st.file_uploader(
                "Upload Macro CSV (single column)",
                type=["csv"],
                key="macro_upload"
            )

            st.markdown("### Differencing")
            diff_order = st.selectbox(
                "Differencing Order",
                [0,1,2,3],
                key="diff_tab1"
            )
            st.divider()
            st.caption("Version 1.0")
            st.caption("Internal Use Only")
    else:
        file = None

    st.header("Macro Time Series Forecast")

    if file:

        df = pd.read_csv(file)
        series = df.iloc[:,0].dropna().reset_index(drop=True)

        st.subheader("Data Preview")
        st.dataframe(series)

        fig, ax = plt.subplots()
        ax.plot(series)
        ax.set_title("Original Series")
        st.pyplot(fig)

        # ---------------- ADF ----------------
        st.divider()
        st.subheader("ADF Test")
        adf_result = adfuller(series)
        st.write("ADF Statistic:", fmt6(adf_result[0]))
        st.write("p-value:", fmt6(adf_result[1]))

        if adf_result[1] < 0.05:
                st.success("Series is Stationary")
        else:
                st.warning("Series is Non-stationary")

        # ---------------- KPSS ----------------
        st.divider()
        st.subheader("KPSS Test")

        try:
            kpss_stat, kpss_p, _, _ = kpss(series, regression='c')

            st.write("KPSS Statistic:", fmt6(kpss_stat))
            st.write("p-value:", fmt6(kpss_p))

            if kpss_p > 0.05:
                st.success("KPSS: Series is Stationary")
            else:
                st.warning("KPSS: Series is Non-stationary")

        except:
            st.warning("KPSS Test not valid for this sample size")

        # ---------------- Differencing ----------------

        if diff_order > 0:
            st.divider()
            st.subheader("Differencing")

            series_used = series.diff(diff_order).dropna().reset_index(drop=True)

            # Plot differenced
            fig, ax = plt.subplots()
            ax.plot(series_used)
            ax.set_title(f"Differenced Series (order={diff_order})")
            st.pyplot(fig)

            # ADF ulang
            st.subheader("ADF Test After Differencing")

            adf_diff = adfuller(series_used)

            st.write("ADF Statistic:", fmt6(adf_diff[0]))
            st.write("p-value:", fmt6(adf_diff[1]))

            if adf_diff[1] < 0.05:
                st.success("Differenced series is Stationary")
            else:
                st.warning("Differenced series is STILL Non-stationary")

        else:
            series_used = series

        # ---------------- ACF PACF ----------------
        st.divider()
        st.subheader("ACF & PACF")

        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_acf(series_used,lags=min(20,len(series_used)//2))
            st.pyplot(fig1)
        with col2:
            fig2 = plot_pacf(series_used,lags=min(20,len(series_used)//2))
            st.pyplot(fig2)

        # ---------------- Parameters ----------------
        st.divider()
        st.subheader("SARIMA Parameters")

        col1,col2,col3=st.columns(3)
        p=col1.number_input("p",0,5,1,key="sarima_p_tab1")
        d=col1.number_input("d",0,3,diff_order,key="sarima_d_tab1")
        q=col1.number_input("q",0,5,1,key="sarima_q_tab1")

        P=col2.number_input("P",0,5,0,key="sarima_P_tab1")
        D=col2.number_input("D",0,3,0,key="sarima_D_tab1")
        Q=col2.number_input("Q",0,5,0,key="sarima_Q_tab1")

        s=col3.number_input("Seasonal Period",0,24,0,key="sarima_s_tab1")

        st.divider()
        st.subheader("ETS Parameters")
        trend = st.selectbox("Trend",["add","mul",None],key="ets_trend_tab1")
        seasonal = st.selectbox("Seasonal",["add","mul",None],key="ets_seasonal_tab1")
        seasonal_period = st.number_input("Seasonal Period ETS",0,24,0,key="ets_period_tab1")

        forecast_period = st.number_input("Forecast Period",1,50,6,key="forecast_tab1")

        # ---------------- Run Models ----------------
        if st.button("Run All Forecast Models", key="run_tab1"):

            def calculate_metrics(actual, fitted):
                actual = np.array(actual)
                fitted = np.array(fitted)

                min_len = min(len(actual), len(fitted))
                actual = actual[:min_len]
                fitted = fitted[:min_len]

                error = actual - fitted

                rmse = np.sqrt(np.mean(error**2))
                mae = np.mean(np.abs(error))
                mape = np.mean(np.abs(error / actual)) * 100

                return rmse, mae, mape

            def ljung_box(residuals):
                lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
                return lb["lb_pvalue"].iloc[0]

            results = {}

            # ================= AUTO ARIMA =================
            auto_model = auto_arima(series, seasonal=False, suppress_warnings=True)
            auto_fitted = auto_model.predict_in_sample()
            auto_forecast = auto_model.predict(n_periods=forecast_period)

            auto_rmse, auto_mae, auto_mape = calculate_metrics(series, auto_fitted)
            auto_lb = ljung_box(auto_model.resid())

            results["Auto ARIMA"] = {
                "model": auto_model,
                "fitted": auto_fitted,
                "forecast": auto_forecast,
                "AIC": auto_model.aic(),
                "BIC": auto_model.bic(),
                "RMSE": auto_rmse,
                "MAE": auto_mae,
                "MAPE": auto_mape,
                "LB": auto_lb
            }

            # ================= SARIMA =================
            sarima_model = SARIMAX(
                series,
                order=(p,d,q),
                seasonal_order=(P,D,Q,s)
            ).fit()

            sarima_fitted = sarima_model.fittedvalues
            sarima_forecast = sarima_model.get_forecast(
                steps=forecast_period
            ).predicted_mean

            sarima_rmse, sarima_mae, sarima_mape = calculate_metrics(series, sarima_fitted)
            sarima_lb = ljung_box(sarima_model.resid)

            results["SARIMA"] = {
                "model": sarima_model,
                "fitted": sarima_fitted,
                "forecast": sarima_forecast,
                "AIC": sarima_model.aic,
                "BIC": sarima_model.bic,
                "RMSE": sarima_rmse,
                "MAE": sarima_mae,
                "MAPE": sarima_mape,
                "LB": sarima_lb
            }

            # ================= ETS =================
            ets_model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_period
            ).fit()

            ets_fitted = ets_model.fittedvalues
            ets_forecast = ets_model.forecast(forecast_period)

            ets_rmse, ets_mae, ets_mape = calculate_metrics(series, ets_fitted)
            ets_lb = ljung_box(series - ets_fitted)

            results["ETS"] = {
                "model": ets_model,
                "fitted": ets_fitted,
                "forecast": ets_forecast,
                "AIC": ets_model.aic,
                "BIC": ets_model.bic,
                "RMSE": ets_rmse,
                "MAE": ets_mae,
                "MAPE": ets_mape,
                "LB": ets_lb
            }

            # =====================================================
            # MODEL COMPARISON (FIXED)
            # =====================================================

            st.divider()
            st.subheader("Model Comparison")

            comp_table = pd.DataFrame([
                {
                    "Model": name,
                    "AIC": res["AIC"],
                    "BIC": res["BIC"],
                    "RMSE": res["RMSE"],
                    "MAE": res["MAE"],
                    "MAPE": res["MAPE"],
                    "Ljung-Box p-value": res["LB"]
                }
                for name, res in results.items()
            ])

            comp_table = comp_table.sort_values("RMSE")

            st.dataframe(comp_table)

            best_model = comp_table.iloc[0]["Model"]

            col1, col2 = st.columns(2)
            col1.metric("Best Model", best_model)
            col2.metric("Best RMSE", fmt4(comp_table.iloc[0]["RMSE"]))

            # =====================================================
            # RESIDUAL DIAGNOSTIC - BEST MODEL
            # =====================================================

            st.divider()
            st.subheader("Residual Diagnostic - Best Model")

            best_model_obj = results[best_model]["model"]

            # Ambil residual
            if best_model == "Auto ARIMA":
                residuals = best_model_obj.resid()
            elif best_model == "SARIMA":
                residuals = best_model_obj.resid
            else:  # ETS
                residuals = series - best_model_obj.fittedvalues

            residuals = pd.Series(residuals).dropna()

            col1, col2 = st.columns(2)

            # Residual vs Fitted
            with col1:
                fig, ax = plt.subplots()
                ax.scatter(range(len(residuals)), residuals)
                ax.axhline(0, linestyle="--")
                ax.set_title("Residual Plot")
                st.pyplot(fig)

            # Histogram
            with col2:
                fig, ax = plt.subplots()
                ax.hist(residuals, bins=10)
                ax.set_title("Residual Histogram")
                st.pyplot(fig)

            # QQ Plot
            fig = sm.qqplot(residuals, line="s")
            st.pyplot(fig)

            # =====================================================
            # COMBINED PLOT
            # =====================================================

            st.divider()
            st.subheader("Combined Plot")

            fig, ax = plt.subplots()
            ax.plot(series, label="Actual", linewidth=3)

            colors = ["red", "green", "purple"]
            i = 0

            for name, res in results.items():
                ax.plot(res["fitted"], label=f"{name} Fitted", color=colors[i])
                ax.plot(
                    range(len(series), len(series)+len(res["forecast"])),
                    res["forecast"],
                    linestyle="dashed",
                    color=colors[i],
                    label=f"{name} Forecast"
                )
                i += 1

            ax.legend()
            st.pyplot(fig)

            # =====================================================
            # MODEL SUMMARIES
            # =====================================================

            st.divider()
            st.subheader("Model Summaries")

            for name, res in results.items():
                with st.expander(f"{name} Summary"):
                    st.text(res["model"].summary())                

# =========================================================
# ========================= TAB 2 =========================
# =========================================================

with tab2:

    st.header("PD Forecast Using MEV (OLS)")

    # =====================================================
    # Upload
    # =====================================================

    if active_tab == "📊 PD MEV Forecast":

        with st.sidebar:
            st.subheader("📊 PD Stress Testing Controls")

            mev_file = st.file_uploader("Upload MEV CSV", key="mev_tab2")
            logit_file = st.file_uploader("Upload Logit CSV", key="logit_tab2")

            normalize = st.checkbox(
                "Normalize Independent Variables (StandardScaler)"
            )
            st.divider()
            st.caption("Version 1.0")
            st.caption("Internal Use Only")
    else:
        mev_file = None
        logit_file = None

    if mev_file and logit_file:

        mev = pd.read_csv(mev_file)
        logit = pd.read_csv(logit_file)

        # =====================================================
        # DEFAULT CONFIG
        # =====================================================

        default_y = "PD - 3"

        y_var = st.selectbox(
            "Dependent Variable",
            logit.columns,
            index=list(logit.columns).index(default_y)
            if default_y in logit.columns else 0
        )

        X_vars = st.multiselect(
            "Independent Candidates",
            mev.columns,
            default=list(mev.columns)[:8]
        )

        # =====================================================
        # SOURCE MAP
        # =====================================================

        source_map = {
            col: [c for c in mev.columns if col in c and c != col]
            for col in mev.columns
        }

        # =====================================================
        # FUNCTIONS
        # =====================================================

        def is_valid_combination(vars_selected, source_map):
            for source, derived_list in source_map.items():
                used = [v for v in vars_selected if v == source or v in derived_list]
                if len(used) > 1:
                    return False
            return True

        def compute_vif(X):
            vif = pd.DataFrame()
            vif["variable"] = X.columns
            vif["VIF"] = [
                variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])
            ]
            return vif

        def check_sign(coef_dict):
            for var, coef in coef_dict.items():
                if var == 'const':
                    continue
                if var not in expected_sign:
                    continue
                if expected_sign[var] == '+' and coef <= 0:
                    return 0
                if expected_sign[var] == '-' and coef >= 0:
                    return 0
            return 1

        # =====================================================
        # PREP DATA
        # =====================================================

        mev_select = mev[X_vars].iloc[3:, :]
        logit_select = logit.iloc[3:, :]

        if normalize:
            scaler = StandardScaler()
            X_select = pd.DataFrame(
                scaler.fit_transform(mev_select),
                index=mev_select.index,
                columns=mev_select.columns
            )
        else:
            X_select = mev_select.copy()

        # =====================================================
        # EXPECTED SIGN
        # =====================================================

        expected_sign = {
            'KURS': '+',
            'PDB': '-',
            'INFLASI': '+',
            'UNEMPLOYEE': '+',
            'BI_RATE': '+',
            'CPI': '+',
            'INDEX_HOUSE': '-',
            'CREDIT_GROWTH': '-',
            'yoy cpi': '+',
            'CREDIT_GROWTH_lag': '-',
            'ln(kurs)': '+', 
            'unemployee_delta': '+', 
            'yoy cpi': '+', 
            'ln(index_house)': '-', 
            'yoy index house': '-',
            'ln(credit_growth)': '-', 
            'sqrt(credit_growth)': '-', 
            'ln(yoy cpi)': '+',
            'INFLASI_lag': '+', 
            'CPI_lag': '+', 
            'CREDIT_GROWTH_lag': '-',
            'INDEX_HOUSE_lag3': '-', 
            'CREDIT_GROWTH_lag3': '-', 
            'UNEMPLOYEE_lag3': '+',
            'ln(UNEMPLOYEE)': '+',  
            'unemployee_delta_2': '+', 
            'UNEMPLOYEE_2': '+',
            'CPI_2': '+', 
            'INFLASI_2': '+', 
            'CREDIT_GROWTH_2': '-', 
            'CPI_CREDIT_GROWTH': '-'
        }

        # =====================================================
        # A. CANDIDATE MODEL SELECTION
        # =====================================================

        st.divider()
        st.subheader("Candidate Model Selection")

        if st.button("Run Candidate Model Selection", key="run_candidate"):

            results = []

            for k in range(1, 5):
                for combo in itertools.combinations(X_vars, k):

                    if not is_valid_combination(combo, source_map):
                        continue

                    try:
                        X = X_select[list(combo)].dropna()
                        y = logit_select.loc[X.index, y_var]

                        X_const = sm.add_constant(X)
                        model = sm.OLS(y, X_const).fit()

                        resid = model.resid

                        reset_test = linear_reset(model, power=2, use_f=True)
                        shapiro_p = shapiro(resid)[1]
                        bp_pvalue = het_breuschpagan(resid, X_const)[1]
                        lb_pvalue = acorr_ljungbox(resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]

                        vif_max = compute_vif(X)

                        y_hat = model.predict(X_const)
                        rmse = np.sqrt(mean_squared_error(y, y_hat))
                        mape = np.mean(np.abs((y - y_hat) / y)) * 100

                        results.append({
                            "model_id": f"M_{len(results)+1}",
                            "dependent": y_var,
                            "independent": ", ".join(combo),
                            "n_var": len(combo),
                            "r2": model.rsquared,
                            "adj_r2": model.rsquared_adj,
                            "f_pvalue": model.f_pvalue,
                            "rmse": rmse,
                            "mape": mape,
                            "reset_test": reset_test.pvalue,
                            "shapiro_p": shapiro_p,
                            "bp_pvalue": bp_pvalue,
                            "ljungbox_p": lb_pvalue,
                            "max_vif": vif_max.to_dict(),
                            "coef": model.params.to_dict(),
                            "std_err": model.bse.to_dict(),
                            "p_value": model.pvalues.to_dict()
                        })

                    except:
                        continue

            result_df = pd.DataFrame(results)

            # Status flags
            result_df['reset_test_status'] = np.where(result_df['reset_test'] > 0.05, 1, 0)
            result_df['shapiro_p_status'] = np.where(result_df['shapiro_p'] > 0.05, 1, 0)
            result_df['bp_pvalue_status'] = np.where(result_df['bp_pvalue'] > 0.05, 1, 0)
            result_df['ljungbox_p_status'] = np.where(result_df['ljungbox_p'] > 0.05, 1, 0)

            result_df['max_vif_status'] = result_df['max_vif'].apply(
                lambda x: 1 if all(v < 10 for v in x['VIF'].values()) else 0
            )

            result_df['pvalue_status'] = result_df['p_value'].apply(
                lambda x: 1 if all(v < 0.05 for v in x.values()) else 0
            )
            result_df['coef_sign_status'] = result_df['coef'].apply(check_sign)

            result_df['model_pass'] = (
                result_df['reset_test_status'] *
                result_df['shapiro_p_status'] *
                result_df['bp_pvalue_status'] *
                result_df['ljungbox_p_status'] *
                result_df['max_vif_status'] *
                result_df['pvalue_status'] *
                result_df['coef_sign_status']
            )

            result_df = result_df.sort_values(
                by=["model_pass","adj_r2","rmse","n_var"],
                ascending=[False, False, False, True]
            )

            st.dataframe(result_df)

        # =====================================================
        # B. RUN SELECTED MODEL (MANUAL)
        # =====================================================

        st.divider()
        st.subheader("Run Selected Model (Manual)")

        selected_vars = st.multiselect(
            "Select Independent Variables",
            X_vars,
            key="manual_vars"
        )

        # =====================================================
        # BUTTON: FIT & SAVE MODEL
        # =====================================================

        if st.button("Run Final Model", key="run_final"):

            if len(selected_vars) == 0:
                st.warning("Please select independent variables first.")
            else:

                X_manual = X_select[selected_vars].dropna()
                y_manual = logit_select.loc[X_manual.index, y_var]

                X_full = sm.add_constant(X_manual)
                model_full = sm.OLS(y_manual, X_full).fit()

                # SAVE TO SESSION
                st.session_state.model_full = model_full
                st.session_state.X_manual = X_manual
                st.session_state.y_manual = y_manual
                st.session_state.selected_vars_manual = selected_vars.copy()

                st.success("Model successfully stored in session.")

        # =====================================================
        # DISPLAY MODEL (OUTSIDE BUTTON)
        # =====================================================

        if "model_full" in st.session_state:

            model_full = st.session_state.model_full
            X_manual = st.session_state.X_manual
            selected_vars = st.session_state.selected_vars_manual
            y_manual = logit_select.loc[X_manual.index, y_var]

            resid = model_full.resid
            X_full = sm.add_constant(X_manual)

            # ===============================
            # STATISTICS
            # ===============================
            reset_test = linear_reset(model_full, power=2, use_f=True)
            shapiro_p = shapiro(resid)[1]
            bp_pvalue = het_breuschpagan(resid, X_full)[1]
            lb_pvalue = acorr_ljungbox(resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]

            r2 = model_full.rsquared
            adj_r2 = model_full.rsquared_adj
            f_pvalue = model_full.f_pvalue

            vif_max = compute_vif(X_manual)

            y_hat = model_full.predict(X_full)
            rmse = np.sqrt(mean_squared_error(y_manual, y_hat))
            mape = np.mean(np.abs((y_manual - y_hat) / y_manual)) * 100

            # =================================================
            # MODEL SUMMARY
            # =================================================
            with st.expander("📘 Model Summary", expanded=False):
                st.text(model_full.summary())

            # =================================================
            # DIAGNOSTICS
            # =================================================
            with st.expander("📊 Diagnostics", expanded=False):

                st.write("R2:", r2)
                st.write("Adj R2:", adj_r2)
                st.write("F-test p-value:", f_pvalue)

                st.write("RESET p-value:", reset_test.pvalue)
                st.write("Shapiro p-value:", shapiro_p)
                st.write("Breusch-Pagan p-value:", bp_pvalue)
                st.write("Ljung-Box p-value:", lb_pvalue)

                st.write("Condition Number:", model_full.condition_number)

                if model_full.condition_number > 100:
                    st.warning("High Condition Number → Potential Multicollinearity Issue")

            # =================================================
            # VIF
            # =================================================
            with st.expander("📈 Multicollinearity (VIF)", expanded=False):
                st.dataframe(vif_max)

            # =================================================
            # RESIDUAL DIAGNOSTIC
            # =================================================
            with st.expander("📉 Residual Diagnostic", expanded=False):

                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots()
                    ax.scatter(model_full.fittedvalues, resid)
                    ax.axhline(0, linestyle="--")
                    ax.set_title("Residual vs Fitted")
                    st.pyplot(fig)

                with col2:
                    fig, ax = plt.subplots()
                    ax.hist(resid, bins=10)
                    ax.set_title("Residual Histogram")
                    st.pyplot(fig)

                fig = sm.qqplot(resid, line="s")
                st.pyplot(fig)

            # =================================================
            # PERFORMANCE
            # =================================================
            with st.expander("⚡ Performance", expanded=False):
                st.write("RMSE:", rmse)
                st.write("MAPE:", mape)

            # =================================================
            # EXECUTIVE SUMMARY
            # =================================================
            with st.expander("🧠 Executive Insight", expanded=False):

                insight_text = ""

                if adj_r2 > 0.7:
                    insight_text += "Model explains majority of PD variation.\n"
                else:
                    insight_text += "Model explanatory power is moderate.\n"

                if reset_test.pvalue > 0.05:
                    insight_text += "No functional form misspecification detected.\n"
                else:
                    insight_text += "Potential functional form issue detected.\n"

                if bp_pvalue > 0.05:
                    insight_text += "No heteroskedasticity detected.\n"
                else:
                    insight_text += "Heteroskedasticity detected.\n"

                if model_full.condition_number > 100:
                    insight_text += "Multicollinearity risk present.\n"
                else:
                    insight_text += "Multicollinearity within acceptable range.\n"

                st.write(insight_text)

            # =================================================
            # SCENARIO SIMULATION
            # =================================================
            with st.expander("🚀 Scenario Simulation", expanded=True):

                params = model_full.params
                latest_X = X_manual.iloc[-1].copy()
                shocked_X = latest_X.copy()

                st.subheader("Baseline Values (Last Observation)")
                st.dataframe(latest_X)

                st.subheader("Apply Shock (%)")

                for var in selected_vars:
                    shock = st.number_input(
                        f"Shock for {var} (%)",
                        min_value=-100.0,
                        max_value=100.0,
                        value=0.0,
                        step=0.5,
                        key=f"shock_{var}"
                    )
                    shocked_X[var] = latest_X[var] * (1 + shock / 100)

                baseline = 0.0
                shocked = 0.0

                if "const" in params:
                    baseline += params["const"]
                    shocked += params["const"]

                for var in selected_vars:
                    if var in params:
                        baseline += params[var] * latest_X[var]
                        shocked += params[var] * shocked_X[var]

                # ===== LOGIT RESULT =====
                st.markdown("### 📊 Logit Projection")

                col1, col2 = st.columns(2)
                col1.metric("Baseline Logit", round(baseline, 6))
                col2.metric(
                    "Shocked Logit",
                    round(shocked, 6),
                    delta=round(shocked - baseline, 6)
                )

                # ===== PD TRANSFORMATION =====
                def logistic_transform(x):
                    return np.exp(x) / (1 + np.exp(x))

                baseline_pd = logistic_transform(baseline)
                shocked_pd = logistic_transform(shocked)

                st.divider()
                st.markdown("### 📉 PD Projection")

                col1, col2 = st.columns(2)
                col1.metric("Baseline PD", round(baseline_pd, 6))
                col2.metric(
                    "Shocked PD",
                    round(shocked_pd, 6),
                    delta=round(shocked_pd - baseline_pd, 6)
                )

            # =================================================
            # Multi Scenario
            # =================================================
            with st.expander("📊 Multi Scenario Grid", expanded=False):

                scenario_shocks = [-10, -5, 0, 5, 10]

                scenario_results = []

                for shock in scenario_shocks:

                    shocked_val = 0
                    if "const" in params:
                        shocked_val += params["const"]

                    for var in selected_vars:
                        shocked_val += params[var] * (latest_X[var] * (1 + shock/100))

                    pd_val = np.exp(shocked_val)/(1+np.exp(shocked_val))

                    scenario_results.append({
                        "Shock %": shock,
                        "PD": pd_val
                    })

                scenario_df = pd.DataFrame(scenario_results)
                st.dataframe(scenario_df)

            # =================================================
            # Export PDF
            # =================================================
            with st.expander("📄 Export PDF Report", expanded=False):

                if st.button("Generate PDF Report"):

                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer)
                    elements = []

                    styles = getSampleStyleSheet()
                    elements.append(Paragraph("PD Model Report", styles["Heading1"]))
                    elements.append(Spacer(1, 0.3*inch))

                    elements.append(Paragraph(f"Adj R2: {round(adj_r2,4)}", styles["Normal"]))
                    elements.append(Paragraph(f"RMSE: {round(rmse,4)}", styles["Normal"]))
                    elements.append(Paragraph(f"MAPE: {round(mape,4)}", styles["Normal"]))

                    doc.build(elements)

                    st.download_button(
                        "Download Report",
                        buffer.getvalue(),
                        file_name="pd_model_report.pdf",
                        mime="application/pdf"
                    )

            st.divider()
            st.markdown("## 🏛 Model Governance Snapshot")

            gov_col1, gov_col2, gov_col3 = st.columns(3)

            gov_col1.metric("Variables Used", len(selected_vars))
            gov_col2.metric("Sample Size", len(X_manual))
            gov_col3.metric("Model Pass Basic Tests",
                            "Yes" if (reset_test.pvalue>0.05 and bp_pvalue>0.05) else "Review Needed")
