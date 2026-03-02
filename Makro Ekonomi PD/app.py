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

from statsmodels.tsa.stattools import adfuller
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

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(layout="wide")
st.title("Credit Risk Modelling Application")

tab1, tab2 = st.tabs([
    "Tab 1 - Macro Forecast",
    "Tab 2 - PD Forecast (MEV OLS)"
])

# =========================================================
# ========================= TAB 1 =========================
# =========================================================

with tab1:

    st.header("Macro Time Series Forecast")

    file = st.file_uploader("Upload Macro CSV (single column)", type=["csv"], key="macro_upload")

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
        st.subheader("ADF Test")
        adf_result = adfuller(series)
        st.write("ADF Statistic:", round(adf_result[0],6))
        st.write("p-value:", round(adf_result[1],6))

        if adf_result[1] < 0.05:
                st.success("Series is Stationary")
        else:
                st.warning("Series is Non-stationary")

        # ---------------- Differencing ----------------
        st.subheader("Differencing")

        diff_order = st.selectbox(
            "Differencing Order",
            [0,1,2,3],
            key="diff_tab1"
        )

        if diff_order > 0:

            series_used = series.diff(diff_order).dropna().reset_index(drop=True)

            # Plot differenced
            fig, ax = plt.subplots()
            ax.plot(series_used)
            ax.set_title(f"Differenced Series (order={diff_order})")
            st.pyplot(fig)

            # ADF ulang
            st.subheader("ADF Test After Differencing")

            adf_diff = adfuller(series_used)

            st.write("ADF Statistic:", round(adf_diff[0],6))
            st.write("p-value:", round(adf_diff[1],6))

            if adf_diff[1] < 0.05:
                st.success("Differenced series is Stationary")
            else:
                st.warning("Differenced series is STILL Non-stationary")

        else:
            series_used = series

        # ---------------- ACF PACF ----------------
        st.subheader("ACF & PACF")

        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_acf(series_used,lags=min(20,len(series_used)//2))
            st.pyplot(fig1)
        with col2:
            fig2 = plot_pacf(series_used,lags=min(20,len(series_used)//2))
            st.pyplot(fig2)

        # ---------------- Parameters ----------------
        st.subheader("SARIMA Parameters")

        col1,col2,col3=st.columns(3)
        p=col1.number_input("p",0,5,1,key="sarima_p_tab1")
        d=col1.number_input("d",0,3,diff_order,key="sarima_d_tab1")
        q=col1.number_input("q",0,5,1,key="sarima_q_tab1")

        P=col2.number_input("P",0,5,0,key="sarima_P_tab1")
        D=col2.number_input("D",0,3,0,key="sarima_D_tab1")
        Q=col2.number_input("Q",0,5,0,key="sarima_Q_tab1")

        s=col3.number_input("Seasonal Period",0,24,0,key="sarima_s_tab1")

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
            st.success(f"Best Model: {best_model}")

            # =====================================================
            # COMBINED PLOT
            # =====================================================

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

            st.subheader("Model Summaries")

            for name, res in results.items():
                st.subheader(name)

                if name == "ETS":
                    st.write("Parameters:")
                    st.write(res["model"].params)
                else:
                    st.text(res["model"].summary())

# =========================================================
# ========================= TAB 2 =========================
# =========================================================

with tab2:

    st.header("PD Forecast Using MEV (OLS)")

    import itertools
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import (
        het_breuschpagan,
        linear_reset,
        acorr_ljungbox
    )
    from scipy.stats import shapiro

    # =====================================================
    # Upload
    # =====================================================

    col1, col2 = st.columns(2)

    mev_file = col1.file_uploader("Upload MEV CSV", key="mev_tab2")
    logit_file = col2.file_uploader("Upload Logit CSV", key="logit_tab2")

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

        normalize = st.checkbox("Normalize Independent Variables (StandardScaler)")

        # =====================================================
        # SOURCE MAP (DEFAULT — bisa Anda modifikasi)
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
            # kalau variable tidak ada di expected_sign → skip
                if var not in expected_sign:
                    continue
                if expected_sign[var] == '+' and coef <= 0:
                    return 0
                if expected_sign[var] == '-' and coef >= 0:
                    return 0
            return 1
        
        # =====================================================
        # ILLOC 3
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
            mean_X = pd.Series(scaler.mean_, index=X_select.columns)
            std_X  = pd.Series(scaler.scale_, index=X_select.columns)
        else:
            X_select = mev_select.copy()

        # =====================================================
        # EXPECTED SIGN (DEFAULT)
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

                        # Statistik
                        r2 = model.rsquared
                        adj_r2 = model.rsquared_adj
                        f_pvalue = model.f_pvalue

                        resid = model.resid

                        reset_test = linear_reset(model, power=2, use_f=True)
                        shapiro_p = shapiro(resid)[1]
                        bp_pvalue = het_breuschpagan(resid, X_const)[1]
                        lb_pvalue = acorr_ljungbox(
                            resid,
                            lags=[4],
                            return_df=True
                        )["lb_pvalue"].iloc[0]

                        vif_max = compute_vif(X)

                        y_hat = model.predict(X_const)
                        rmse = np.sqrt(mean_squared_error(y, y_hat))
                        mape = np.mean(np.abs((y - y_hat) / y)) * 100

                        results.append({
                            "model_id": f"M_{len(results)+1}",
                            "dependent": y_var,
                            "independent": ", ".join(combo),
                            "n_var": len(combo),
                            "r2": r2,
                            "adj_r2": adj_r2,
                            "f_pvalue": f_pvalue,
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

            st.subheader("Candidate Model Ranking")
            st.dataframe(result_df)

        # =====================================================
        # B. RUN SELECTED MODEL (MANUAL)
        # =====================================================

        st.header("Run Selected Model (Manual)")

        selected_vars = st.multiselect(
            "Select Independent Variables",
            X_vars,
            key="manual_vars"
        )

        if st.button("Run Final Model", key="run_final"):

            X = X_select[selected_vars].dropna()
            y = logit_select.loc[X.index, y_var]

            X_full = sm.add_constant(X)
            model_full = sm.OLS(y, X_full).fit()

            st.subheader("Model Summary")
            st.text(model_full.summary())

            reset_test = linear_reset(model_full, power=2, use_f=True)
            resid = model_full.resid

            r2 = model_full.rsquared
            adj_r2 = model_full.rsquared_adj
            f_pvalue = model_full.f_pvalue

            shapiro_p = shapiro(resid)[1]
            bp_pvalue = het_breuschpagan(resid, X_full)[1]
            lb_pvalue = acorr_ljungbox(resid, lags=[4], return_df=True)["lb_pvalue"].iloc[0]

            vif_max = compute_vif(X)

            y_hat = model_full.predict(X_full)
            rmse = np.sqrt(mean_squared_error(y, y_hat))
            mape = np.mean(np.abs((y - y_hat) / y)) * 100

            st.subheader("Diagnostics")

            st.write("R2:", r2)
            st.write("Adj R2:", adj_r2)
            st.write("F-test p-value:", f_pvalue)
            st.write("RESET p-value:", reset_test.pvalue)
            st.write("Shapiro p-value:", shapiro_p)
            st.write("Breusch-Pagan p-value:", bp_pvalue)
            st.write("Ljung-Box p-value:", lb_pvalue)

            st.subheader("VIF")
            st.dataframe(vif_max)

            st.subheader("Performance")
            st.write("RMSE:", rmse)
            st.write("MAPE:", mape)
