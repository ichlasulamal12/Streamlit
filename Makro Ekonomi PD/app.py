# =========================================================
# CREDIT RISK MODELLING APPLICATION
# =========================================================

import hashlib
import time
import os
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import io
from datetime import datetime
import pickle

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="PD Forecast System",
                   page_icon="https://raw.githubusercontent.com/ichlasulamal12/Streamlit/main/Makro Ekonomi PD/logo.png",
                   layout="wide")

# =========================================================
# LOGIN CONFIG
# =========================================================

USERS = {
    "admin": hashlib.sha256("adm123_A1!".encode()).hexdigest(),
    "riskuser": hashlib.sha256("risk123_A1!".encode()).hexdigest()
}

SESSION_TIMEOUT = 900
LOG_FILE = "user_activity_log.csv"

# =========================================================
# SESSION INIT
# =========================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

# =========================================================
# LOGIN STYLE
# =========================================================

if not st.session_state.authenticated:

    st.markdown("""
    <style>

    [data-testid="stSidebar"] {display:none;}
    header {visibility:hidden;}

    .stApp {
        background-color:#f4f6f9;
    }

    .login-box{
        background:white;
        padding:15px;
        border-radius:12px;
        box-shadow:0px 8px 25px rgba(0,0,0,0.12);
        max-width:520px;
        margin:auto;
    }

    .login-title{
        text-align:center;
        font-size:26px;
        font-weight:600;
        margin-bottom:20px;
    }

    .login-footer{
        text-align:center;
        font-size:12px;
        color:#888;
        margin-top:15px;
    }

    </style>
    """, unsafe_allow_html=True)

# =========================================================
# USER LOGGING
# =========================================================

def log_user_action(username, action):

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"{timestamp},{username},{action}\n"

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,username,action\n")

    with open(LOG_FILE, "a") as f:
        f.write(log_entry)

# =========================================================
# LOGIN PAGE
# =========================================================

def login_page():

    col1, col2, col3 = st.columns([1,4,1])

    with col2:
        st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://raw.githubusercontent.com/ichlasulamal12/Streamlit/main/Makro Ekonomi PD/logo.png" width="110">
        </div>
        """,
        unsafe_allow_html=True
        )       

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")


        if st.button("Login", use_container_width=True):

            if username in USERS:

                hashed_input = hashlib.sha256(password.encode()).hexdigest()

                if hashed_input == USERS[username]:

                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.last_activity = time.time()

                    log_user_action(username, "Login")

                    st.success("Login successful")

                    st.rerun()

                else:
                    st.error("Invalid credentials")

            else:
                st.error("Invalid credentials")


        st.markdown(
        '<div class="login-footer">Internal Risk Modelling System</div></div>',
        unsafe_allow_html=True
        )

# =========================================================
# SESSION TIMEOUT
# =========================================================

def check_session_timeout():

    current_time = time.time()

    if st.session_state.authenticated:

        if current_time - st.session_state.last_activity > SESSION_TIMEOUT:

            log_user_action(st.session_state.username, "Session Timeout")

            st.session_state.authenticated = False
            st.session_state.username = None

            st.warning("Session expired. Please login again.")

            st.rerun()

        else:
            st.session_state.last_activity = current_time

# =========================================================
# AUTHENTICATION FLOW
# =========================================================

if not st.session_state.authenticated:

    login_page()

    st.stop()

else:

    check_session_timeout()

# =========================================================
# PROFESSIONAL CREDIT RISK THEME
# =========================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>

/* ================= GLOBAL ================= */

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: #1f2937;
    font-size: 14px;
}

/* ================= APP BACKGROUND ================= */

.stApp {
    background-color: #f4f6f9;
}

/* ================= HEADINGS ================= */

h1 {
    color: #111827;
    font-weight: 700;
}

h2, h3 {
    color: #1f2937;
    font-weight: 600;
}

/* ================= METRIC CARDS ================= */

[data-testid="stMetric"] {
    background-color: white;
    border-radius: 10px;
    padding: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}

/* ================= DATAFRAME ================= */

.stDataFrame {
    background-color: white;
    border-radius: 8px;
}

/* ================= EXPANDER ================= */

details {
    background-color: white;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    padding: 8px;
}

/* ================= BUTTON ================= */

button[kind="primary"] {
    background-color: #2563eb !important;
    border-radius: 8px !important;
    color: white !important;
}

button[kind="secondary"] {
    border-radius: 8px !important;
}

/* ================= TABS ================= */

button[data-baseweb="tab"] {
    font-weight: 600;
}

/* ================= DIVIDER ================= */

hr {
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

/* ================= CARD STYLE ================= */

.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.05);
}

/* ================= EXECUTIVE KPI ================= */

.kpi-card {
    background: white;
    border-left: 6px solid #2563eb;
    padding: 15px;
    border-radius: 10px;
}

/* ================= RISK COLORS ================= */

.risk-low {
    color: #16a34a;
    font-weight: 600;
}

.risk-medium {
    color: #f59e0b;
    font-weight: 600;
}

.risk-high {
    color: #dc2626;
    font-weight: 600;
}

/* ================= SCENARIO COLORS ================= */

.scenario-base {
    color: #2563eb;
    font-weight: 600;
}

.scenario-good {
    color: #16a34a;
    font-weight: 600;
}

.scenario-bad {
    color: #dc2626;
    font-weight: 600;
}

/* ================= FORWARD LOOKING BOX ================= */

.forward-box {
    background-color: #eef2ff;
    border-left: 6px solid #4f46e5;
    padding: 14px;
    border-radius: 8px;
}

/* ================= SUCCESS BOX ================= */

.success-box {
    background-color: #ecfdf5;
    border-left: 6px solid #10b981;
    padding: 12px;
    border-radius: 8px;
}

/* ================= WARNING BOX ================= */

.warning-box {
    background-color: #fff7ed;
    border-left: 6px solid #f59e0b;
    padding: 12px;
    border-radius: 8px;
}

/* ================= ERROR BOX ================= */

.error-box {
    background-color: #fef2f2;
    border-left: 6px solid #dc2626;
    padding: 12px;
    border-radius: 8px;
}

/* ================= TABLE IMPROVEMENT ================= */

thead tr th {
    background-color: #f3f4f6 !important;
}

/* ================= MAIN PADDING ================= */

.block-container {
    padding-top: 1.2rem;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# MAIN APP STARTS HERE
# ===============================

st.title("Credit Risk Modelling Application")

st.markdown("""
**Integrated Macro Forecasting & PD MEV Forecast Engine**  
Internal Quantitative Risk Tool
""")

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(f"👤 Logged in as: **{st.session_state.username}**")
    if st.button("Logout"):
        log_user_action(st.session_state.username, "Logout")
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()    
    
    st.divider()
    st.header("📂 Navigation")
    active_tab = st.radio(
        "Select Module",
        ["📈 Macro Forecast", "📊 PD MEV Forecast", "📉 Logit PD MEV Calculation"],
        key="active_module"
    )
    st.divider()

tab1, tab2, tab3 = st.tabs([
    "📈 Macro Forecast",
    "📊 PD MEV Forecast",
    "📉 Logit PD MEV Calculation"
])

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def fmt4(x):
    return round(float(x), 4)

def fmt6(x):
    return round(float(x), 6)

def fmt_pct(x):
    return f"{x*100:.2f}%"

def logistic(x):
    return 1/(1+np.exp(-x))

def good(coef, base_series, shock=0.1):

    return np.where(
        coef > 0,
        np.where(base_series > 0,
                 base_series + (base_series * -shock),
                 base_series + (base_series * shock)),
        np.where(base_series > 0,
                 base_series + (base_series * shock),
                 base_series + (base_series * -shock))
    )

def bad(coef, base_series, shock=0.1):

    return np.where(
        coef > 0,
        np.where(base_series > 0,
                 base_series + (base_series * shock),
                 base_series + (base_series * -shock)),
        np.where(base_series > 0,
                 base_series + (base_series * -shock),
                 base_series + (base_series * shock))
    )

@st.cache_data(show_spinner=False)
def read_pd_actual(file_bytes):

    if file_bytes is None:
        return None

    import io

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="PD%",
        skiprows=5,
        engine="openpyxl"
    )

    return df

# =====================================================
# READ PD RATING
# =====================================================

@st.cache_data(show_spinner=False)
def read_pd_rating(file_bytes):

    if file_bytes is None:
        return None

    import io

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name="Migration PD Rating 2",
        engine="openpyxl"
    )

    return df

def pd_actual_segment(df):

    return (
        df["PD 1.3"].iloc[-1] +
        df["PD 2.3"].iloc[-1]
    )

@st.cache_data(show_spinner=False)
def compute_bankwide_pd(
    pd_rating_mk,
    pd_rating_inv,
    pd_rating_con,
    pd_rating_chn
):

    if any(v is None for v in [
        pd_rating_mk,
        pd_rating_inv,
        pd_rating_con,
        pd_rating_chn
    ]):
        return None

    mk_num = pd_rating_mk.select_dtypes(include=np.number)
    inv_num = pd_rating_inv.select_dtypes(include=np.number)
    con_num = pd_rating_con.select_dtypes(include=np.number)
    chn_num = pd_rating_chn.select_dtypes(include=np.number)

    pd_rating_bw = mk_num + inv_num + con_num + chn_num

    pd_rating_bw['Rating Movement'] = pd_rating_mk['Rating Movement'].copy()

    pd_rating_bw = pd_rating_bw.iloc[:30,6:].dropna()

    pd_rating_bw = pd_rating_bw[
        [pd_rating_bw.columns[-1]] +
        list(pd_rating_bw.columns[:-1])
    ].set_index("Rating Movement")

    pd_rating_bw.loc[1] = pd_rating_bw.iloc[:5].sum()
    pd_rating_bw.loc[2] = pd_rating_bw.iloc[5:10].sum()
    pd_rating_bw.loc[3] = pd_rating_bw.iloc[10:15].sum()
    pd_rating_bw.loc[4] = pd_rating_bw.iloc[15:20].sum()
    pd_rating_bw.loc[5] = pd_rating_bw.iloc[20:25].sum()

    pd_rating_bw.loc["13%"] = pd_rating_bw.iloc[2] / pd_rating_bw.iloc[25]
    pd_rating_bw.loc["14%"] = pd_rating_bw.iloc[3] / pd_rating_bw.iloc[25]
    pd_rating_bw.loc["15%"] = pd_rating_bw.iloc[4] / pd_rating_bw.iloc[25]

    pd_rating_bw.loc["23%"] = pd_rating_bw.iloc[7] / pd_rating_bw.iloc[26]
    pd_rating_bw.loc["24%"] = pd_rating_bw.iloc[8] / pd_rating_bw.iloc[26]
    pd_rating_bw.loc["25%"] = pd_rating_bw.iloc[9] / pd_rating_bw.iloc[26]

    pd_rating_bw.loc["PD-1"] = pd_rating_bw.iloc[30:33].sum()
    pd_rating_bw.loc["PD-2"] = pd_rating_bw.iloc[33:36].sum()

    pd_actual_bw = pd_rating_bw.iloc[-2:].T.iloc[1:]

    pd_actual_bw["PD-1 Smoothing"] = pd_actual_bw["PD-1"]

    pd_actual_bw["PD-2 Smoothing"] = np.where(
        pd_actual_bw["PD-2"] < pd_actual_bw["PD-1"],
        pd_actual_bw["PD-1"],
        pd_actual_bw["PD-2"]
    )

    pd_actual = (
        pd_actual_bw["PD-1 Smoothing"].iloc[-1] +
        pd_actual_bw["PD-2 Smoothing"].iloc[-1]
    )

    return pd_actual

def compute_forward(avg_pd_hat, pd_actual):

    ratio = avg_pd_hat / pd_actual

    if ratio > 1.2:
        return 1.2
    elif ratio < 0.8:
        return 0.8
    else:
        return ratio

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

            log_user_action(st.session_state.username, "Run Macro Forecast")
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
            st.subheader("📊 PD MEV Forecast Controls")

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

            scaler_mean = pd.Series(scaler.mean_, index=mev_select.columns)
            scaler_std  = pd.Series(scaler.scale_, index=mev_select.columns)

        else:
            X_select = mev_select.copy()
            scaler_mean = None
            scaler_std  = None

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

            log_user_action(st.session_state.username, "Run Candidate Model")
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

            log_user_action(st.session_state.username, "Run PD Manual Model")
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
            with st.expander("🚀 Scenario Simulation", expanded=False):

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
                    log_user_action(st.session_state.username, "Scenario Simulation")

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
            # EXPORT TRAINED MODEL
            # =================================================

            with st.expander("💾 Export Model", expanded=False):

                if st.button("Prepare Model File (.pkl)"):

                    model_package = {
                        "model": model_full,
                        "selected_vars": selected_vars,
                        "dependent_var": y_var,
                        "normalize": normalize,
                        "mean": scaler_mean,
                        "std": scaler_std,
                        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                    }

                    model_bytes = pickle.dumps(model_package)

                    st.download_button(
                        label="Download Model (.pkl)",
                        data=model_bytes,
                        file_name="PD_OLS_Model.pkl",
                        mime="application/octet-stream"
                    )

                    st.success("Model ready for download.")            

            # =================================================
            # Export PDF
            # =================================================
            with st.expander("📄 Generate Report", expanded=False):

                log_user_action(st.session_state.username, "Generate PDF Report")
                if st.button("Generate PDF Report"):

                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer)
                    elements = []
                    styles = getSampleStyleSheet()

                    # =====================================================
                    # COVER PAGE
                    # =====================================================

                    elements.append(Paragraph("PD Model Professional Report", styles["Heading1"]))
                    elements.append(Spacer(1, 0.4 * inch))

                    elements.append(Paragraph(
                        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        styles["Normal"]
                    ))
                    elements.append(Spacer(1, 0.2 * inch))

                    elements.append(Paragraph(f"Dependent Variable: {y_var}", styles["Normal"]))
                    elements.append(Paragraph(f"Independent Variables: {', '.join(selected_vars)}", styles["Normal"]))
                    elements.append(Paragraph(f"Sample Size: {len(X_manual)}", styles["Normal"]))
                    elements.append(Spacer(1, 0.5 * inch))

                    elements.append(PageBreak())

                    # =====================================================
                    # EXECUTIVE SUMMARY
                    # =====================================================

                    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    summary_text = ""

                    if adj_r2 > 0.7:
                        summary_text += "Model explains strong variation in PD. "
                    elif adj_r2 > 0.4:
                        summary_text += "Model explains moderate variation in PD. "
                    else:
                        summary_text += "Model explanatory power is limited. "

                    if reset_test.pvalue > 0.05:
                        summary_text += "No functional form misspecification detected. "
                    else:
                        summary_text += "Potential functional form issue detected. "

                    if bp_pvalue > 0.05:
                        summary_text += "No heteroskedasticity detected. "
                    else:
                        summary_text += "Heteroskedasticity present. "

                    if model_full.condition_number > 100:
                        summary_text += "Multicollinearity risk observed."

                    elements.append(Paragraph(summary_text, styles["Normal"]))
                    elements.append(Spacer(1, 0.4 * inch))

                    # =====================================================
                    # MODEL STATISTICS
                    # =====================================================

                    elements.append(Paragraph("Model Statistics", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    stats_data = [
                        ["Metric", "Value"],
                        ["R2", round(r2,6)],
                        ["Adj R2", round(adj_r2,6)],
                        ["F-test p-value", round(f_pvalue,6)],
                        ["Condition Number", round(model_full.condition_number,4)]
                    ]

                    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
                    stats_table.setStyle(TableStyle([
                        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
                        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
                    ]))

                    elements.append(stats_table)
                    elements.append(Spacer(1, 0.4 * inch))

                    # =====================================================
                    # ASSUMPTION TESTS
                    # =====================================================

                    elements.append(Paragraph("Assumption Tests", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    assumption_data = [
                        ["Test", "p-value"],
                        ["RESET", round(reset_test.pvalue,6)],
                        ["Shapiro", round(shapiro_p,6)],
                        ["Breusch-Pagan", round(bp_pvalue,6)],
                        ["Ljung-Box", round(lb_pvalue,6)]
                    ]

                    assumption_table = Table(assumption_data, colWidths=[3*inch,2*inch])
                    assumption_table.setStyle(TableStyle([
                        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
                        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
                    ]))

                    elements.append(assumption_table)
                    elements.append(PageBreak())

                    # =====================================================
                    # PERFORMANCE
                    # =====================================================

                    elements.append(Paragraph("Performance Metrics", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    perf_data = [
                        ["RMSE", round(rmse,6)],
                        ["MAPE (%)", round(mape,6)]
                    ]

                    perf_table = Table(perf_data, colWidths=[3*inch,2*inch])
                    perf_table.setStyle(TableStyle([
                        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
                    ]))

                    elements.append(perf_table)
                    elements.append(Spacer(1, 0.4 * inch))

                    # =====================================================
                    # COEFFICIENT TABLE
                    # =====================================================

                    elements.append(Paragraph("Coefficient Table", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    coef_data = [["Variable","Coef","Std Err","p-value"]]

                    for var in model_full.params.index:
                        coef_data.append([
                            var,
                            round(model_full.params[var],6),
                            round(model_full.bse[var],6),
                            round(model_full.pvalues[var],6)
                        ])

                    coef_table = Table(coef_data)
                    coef_table.setStyle(TableStyle([
                        ('BACKGROUND',(0,0),(-1,0),colors.lightgrey),
                        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
                    ]))

                    elements.append(coef_table)
                    elements.append(PageBreak())

                    # =====================================================
                    # RESIDUAL PLOTS
                    # =====================================================

                    elements.append(Paragraph("Residual Diagnostics", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    # Residual vs Fitted
                    fig, ax = plt.subplots()
                    ax.scatter(model_full.fittedvalues, resid)
                    ax.axhline(0, linestyle="--")
                    ax.set_title("Residual vs Fitted")
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format="png")
                    plt.close(fig)
                    img_buffer.seek(0)
                    elements.append(Image(img_buffer, width=5*inch, height=3*inch))
                    elements.append(Spacer(1, 0.3 * inch))

                    # QQ Plot
                    fig = sm.qqplot(resid, line="s")
                    img_buffer = io.BytesIO()
                    fig.savefig(img_buffer, format="png")
                    plt.close(fig)
                    img_buffer.seek(0)
                    elements.append(Image(img_buffer, width=5*inch, height=3*inch))
                    elements.append(PageBreak())

                    # =====================================================
                    # SCENARIO RESULT
                    # =====================================================

                    elements.append(Paragraph("Scenario Simulation Result", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    scenario_data = [
                        ["Baseline Logit", round(baseline,6)],
                        ["Shocked Logit", round(shocked,6)],
                        ["Baseline PD", round(baseline_pd,6)],
                        ["Shocked PD", round(shocked_pd,6)]
                    ]

                    scenario_table = Table(scenario_data)
                    scenario_table.setStyle(TableStyle([
                        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
                    ]))

                    elements.append(scenario_table)
                    elements.append(Spacer(1, 0.4 * inch))

                    # =====================================================
                    # GOVERNANCE SUMMARY
                    # =====================================================

                    elements.append(Paragraph("Governance Summary", styles["Heading2"]))
                    elements.append(Spacer(1, 0.2 * inch))

                    gov_note = "Model Passed Core Diagnostic Tests."

                    if reset_test.pvalue < 0.05 or bp_pvalue < 0.05:
                        gov_note = "Model Requires Review - Diagnostic Issues Detected."

                    elements.append(Paragraph(gov_note, styles["Normal"]))

                    # =====================================================
                    # BUILD
                    # =====================================================

                    doc.build(elements)

                    st.download_button(
                        "Download Report",
                        buffer.getvalue(),
                        file_name="PD_Model_Report.pdf",
                        mime="application/pdf"
                    )

            st.divider()
            st.markdown("## 🏛 Model Governance Snapshot")

            gov_col1, gov_col2, gov_col3 = st.columns(3)

            gov_col1.metric("Variables Used", len(selected_vars))
            gov_col2.metric("Sample Size", len(X_manual))
            gov_col3.metric("Model Pass Basic Tests",
                            "Yes" if (reset_test.pvalue>0.05 and bp_pvalue>0.05) else "Review Needed")

# =========================================================
# ========================= TAB 3 =========================
# =========================================================

with tab3:

    st.header("Logit PD MEV Calculation Engine")

    if active_tab == "📉 Logit PD MEV Calculation":
        
        with st.sidebar:

            with st.expander("Model and MEV Files", expanded=True):
                model_file = st.file_uploader("Model PKL",type=["pkl"])
                mev_file = st.file_uploader("MEV Forecast",type=["xlsx"])
            with st.expander("Segment Files", expanded=False):
                mk_file = st.file_uploader("Modal Kerja",type=["xlsx"])
                inv_file = st.file_uploader("Investasi",type=["xlsx"])
                con_file = st.file_uploader("Konsumsi",type=["xlsx"])
                chn_file = st.file_uploader("Channeling",type=["xlsx"])
            st.divider()
            st.caption("Version 1.0")
            st.caption("Internal Use Only")
    else:
        model_file = None
        mev_file = None
        mk_file = None
        inv_file = None
        con_file = None
        chn_file = None

    # =====================================================
    # LOAD MODEL
    # =====================================================

    if model_file:
        st.subheader("1. Model")
        model_data = pickle.load(model_file)

        model = model_data["model"]
        mean = model_data["mean"]
        std = model_data["std"]
        selected_vars = model_data["selected_vars"]
        dependent_var = model_data["dependent_var"]

        st.success("Model Loaded Successfully")

        st.write("Dependent:", dependent_var)
        st.write("Independent Variables:", selected_vars)
        
        lag_vars = [v for v in selected_vars if v.strip().lower().startswith("lag")]
        mev_vars = [v for v in selected_vars if v not in lag_vars]

        if len(lag_vars) > 0:
            st.info(f"Lag variable from dependent detected: {lag_vars}")

        # =====================================================
        # MEV FORECAST
        # =====================================================

        st.divider()

        if mev_file:
            st.subheader("2. MEV Forecast")
            mev = pd.read_excel(mev_file)

            st.write("MEV Preview")
            st.dataframe(mev.head())

            mev_forecast = mev.iloc[-12:].reset_index(drop=True)

            # =====================================================
            # NORMALIZATION
            # =====================================================

            st.divider()
            st.subheader("3. Normalization")

            normalize = st.checkbox("Use Model Normalization", value=True)

            # ==============================
            # IF MODEL USES LAG
            # ==============================

            if len(lag_vars) > 0:

                lag_var = lag_vars[0]

                st.divider()
                st.subheader("Initial Lag Logit")

                initial_logit = st.number_input(
                    f"Initial value for {lag_var}",
                    value=0.0
                )

                if normalize:

                    mev_scaled = pd.DataFrame()

                    for var in selected_vars:

                        # skip lag variable (not in MEV file)
                        if var == lag_var:
                            continue

                        if var in mev_forecast.columns:

                            mev_scaled[var] = (
                                mev_forecast[var] - mean[var]
                            ) / std[var]

                        else:

                            st.error(f"Variable {var} not found in MEV file")

                    X_base = mev_scaled.copy()

                    # normalize lag
                    initial_logit_scaled = (
                        initial_logit - mean[lag_var]
                    ) / std[lag_var]

                else:

                    X_base = mev_forecast[
                        [v for v in selected_vars if v != lag_var]
                    ].copy()

                    initial_logit_scaled = initial_logit


            # ==============================
            # IF MODEL HAS NO LAG
            # ==============================

            else:

                if normalize:

                    mev_scaled = pd.DataFrame()

                    for var in selected_vars:

                        if var in mev_forecast.columns:

                            mev_scaled[var] = (
                                mev_forecast[var] - mean[var]
                            ) / std[var]

                        else:

                            st.error(f"Variable {var} not found in MEV file")

                    X_base = mev_scaled.copy()

                else:

                    X_base = mev_forecast[selected_vars].copy()

                initial_logit_scaled = None            
            # =====================================================
            # SCENARIO GENERATION
            # =====================================================

            st.divider()
            st.subheader("4. Scenario Generation")

            scenario = pd.DataFrame()

            for var in mev_vars:

                # skip lag variable (not from MEV)
                if var in lag_vars:
                    continue

                coef = model.params[var]

                base = X_base[var]

                good_s = good(coef, base)
                bad_s = bad(coef, base)

                scenario[f"{var}_base"] = base
                scenario[f"{var}_good"] = good_s
                scenario[f"{var}_bad"] = bad_s

                scenario[f"simulation_{var}_base"] = base*0.8 + good_s*0.1 + bad_s*0.1
                scenario[f"simulation_{var}_good"] = base*0.1 + good_s*0.8 + bad_s*0.1
                scenario[f"simulation_{var}_bad"] = base*0.1 + good_s*0.1 + bad_s*0.8

            st.write("Scenario")
            st.dataframe(scenario)

            # =====================================================
            # MODEL PREDICTION
            # =====================================================

            st.divider()
            st.subheader("5. PD Projection")

            run_projection = st.button("Run PD Projection")

            def predict_pd(prefix):

                if len(lag_vars) == 0:

                    cols = [f"simulation_{v}_{prefix}" for v in mev_vars]

                    X = scenario[cols].copy()
                    X.columns = mev_vars

                    X_const = sm.add_constant(X, has_constant="add")
                    X_const = X_const.reindex(columns=model.params.index, fill_value=0)

                    logit = model.predict(X_const)

                    return logistic(logit)

                else:

                    results = []
                    lag_val = initial_logit_scaled

                    for i in range(len(scenario)):

                        row = {}

                        for var in selected_vars:

                            if var in lag_vars:
                                row[var] = lag_val
                            else:
                                row[var] = scenario[f"simulation_{var}_{prefix}"].iloc[i]

                        X = pd.DataFrame([row])

                        X_const = sm.add_constant(X, has_constant="add")
                        X_const = X_const.reindex(columns=model.params.index, fill_value=0)

                        logit_val = model.predict(X_const)[0]

                        results.append(logit_val)

                        # normalize lag for next iteration
                        if normalize:
                            lag_val = (logit_val - mean[lag_var]) / std[lag_var]
                        else:
                            lag_val = logit_val

                    return logistic(pd.Series(results))

            if run_projection:

                with st.spinner("Running PD Projection..."):

                    pd_base = predict_pd("base")
                    pd_good = predict_pd("good")
                    pd_bad = predict_pd("bad")

                    result_df = pd.DataFrame({
                        "PD_Base": pd_base,
                        "PD_Good": pd_good,
                        "PD_Bad": pd_bad
                    })

                    avg_pd_hat = np.mean([
                        pd_base.iloc[-1],
                        pd_good.iloc[-1],
                        pd_bad.iloc[-1]
                    ])

                    st.session_state.avg_pd_hat = avg_pd_hat
                    st.session_state.result_df = result_df
                    st.session_state.scenario = scenario

            if "result_df" in st.session_state:
                st.dataframe(st.session_state.result_df)

            if "avg_pd_hat" in st.session_state:
                st.metric("Average PD Forecast", round(st.session_state.avg_pd_hat,4))                    

            # =====================================================
            # SEGMENT FILES and FORWARD LOOKING CALCULATION
            # =====================================================

            st.divider()
            st.subheader("6. PD Segment and Forward Looking Factor")

            if "avg_pd_hat" not in st.session_state or "result_df" not in st.session_state:
                st.warning("Please run PD Projection first.")
                st.stop()

            run_calc = st.button("Run Calculation")

            if run_calc:
                st.session_state.segment_ready = True

            # =====================================================
            # MAIN CALCULATION
            # =====================================================

            if st.session_state.get("segment_ready", False):

                with st.spinner("Running calculation..."):

                    pd_mk = read_pd_actual(mk_file.getvalue()) if mk_file else None
                    pd_inv = read_pd_actual(inv_file.getvalue()) if inv_file else None
                    pd_con = read_pd_actual(con_file.getvalue()) if con_file else None
                    pd_chn = read_pd_actual(chn_file.getvalue()) if chn_file else None

                    pd_rating_mk = read_pd_rating(mk_file.getvalue()) if mk_file else None
                    pd_rating_inv = read_pd_rating(inv_file.getvalue()) if inv_file else None
                    pd_rating_con = read_pd_rating(con_file.getvalue()) if con_file else None
                    pd_rating_chn = read_pd_rating(chn_file.getvalue()) if chn_file else None

                    results = []

                    segments = {
                        "Modal Kerja": pd_mk,
                        "Investasi": pd_inv,
                        "Konsumsi": pd_con,
                        "Channeling": pd_chn
                    }

                    avg_pd_hat = st.session_state.avg_pd_hat

                    for seg, df in segments.items():

                        if df is not None:

                            try:

                                pd_actual = pd_actual_segment(df)

                                fl = compute_forward(
                                    avg_pd_hat,
                                    pd_actual
                                )

                                results.append({
                                    "Segment": seg,
                                    "PD Actual": pd_actual,
                                    "PD Forecast": avg_pd_hat,
                                    "Forward Looking Factor": fl
                                })

                            except:
                                continue

                    result_segment = pd.DataFrame(results)

                    # =====================================================
                    # STORE ORIGINAL ONLY ON FIRST RUN
                    # =====================================================

                    if "result_segment_original" not in st.session_state:
                        st.session_state.result_segment_original = result_segment.copy()

                    if "result_segment" not in st.session_state:
                        st.session_state.result_segment = result_segment.copy()

                    # =====================================================
                    # EDITABLE TABLE
                    # =====================================================

                    st.write("Edit PD Actual if needed")

                    edited_segment = st.data_editor(
                        st.session_state.result_segment,
                        use_container_width=True,
                        num_rows="fixed",
                        key="segment_editor"
                    )

                    # =====================================================
                    # RECALCULATE FORWARD LOOKING
                    # =====================================================

                    recalc_results = []

                    for _, row in edited_segment.iterrows():

                        pd_actual = row["PD Actual"]

                        fl = compute_forward(
                            st.session_state.avg_pd_hat,
                            pd_actual
                        )

                        recalc_results.append(fl)

                    edited_segment["Forward Looking Factor"] = recalc_results

                    # SAVE BACK TO SESSION
                    st.session_state.result_segment = edited_segment.copy()

                    st.write("Updated Result")
                    st.dataframe(edited_segment)

                    # =====================================================
                    # RESET BUTTON
                    # =====================================================

                    col1, col2 = st.columns([1,1])

                    with col1:

                        if st.button("Reset to Original"):

                            st.session_state.result_segment = (
                                st.session_state.result_segment_original.copy()
                            )

                            st.rerun()

                    # =====================================================
                    # BANKWIDE
                    # =====================================================

                    col1, col2 = st.columns(2)

                    if len(results) > 0:

                        bankwide_actual = None

                        if (
                            pd_rating_mk is not None and
                            pd_rating_inv is not None and
                            pd_rating_con is not None and
                            pd_rating_chn is not None
                        ):

                            bankwide_actual = compute_bankwide_pd(
                                pd_rating_mk,
                                pd_rating_inv,
                                pd_rating_con,
                                pd_rating_chn
                            )

                            col1.metric(
                                "Bankwide PD Actual",
                                round(bankwide_actual,4)
                            )

                        else:

                            st.warning("Please upload all segment files to compute Bankwide")

                        if bankwide_actual is not None:

                            forward_looking_bw = compute_forward(
                                avg_pd_hat,
                                bankwide_actual
                            )

                            col2.metric(
                                "Bankwide Forward Looking",
                                round(forward_looking_bw,4)
                            )

                    # =====================================================
                    # EXPORT
                    # =====================================================

                    if "result_segment" in st.session_state:

                        st.divider()
                        st.subheader("7. Export Result")

                        buffer = io.BytesIO()

                        with pd.ExcelWriter(buffer) as writer:

                            if "scenario" in st.session_state:
                                st.session_state.scenario.to_excel(
                                    writer,
                                    sheet_name="MEV Impact",
                                    index=False
                                )

                            if "result_df" in st.session_state:
                                st.session_state.result_df.to_excel(
                                    writer,
                                    sheet_name="PD Projection",
                                    index=False
                                )

                            st.session_state.result_segment.to_excel(
                                writer,
                                sheet_name="Segment Result",
                                index=False
                            )

                        st.download_button(
                            "Download Excel Report",
                            buffer.getvalue(),
                            "PD_MEV_Result.xlsx"
                        )
