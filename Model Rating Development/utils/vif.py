import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(df):

    X = df.copy()

    # drop non numeric
    X = X.select_dtypes(include=["number"])

    vif_data = []

    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X.values, i)

        vif_data.append({
            "variable": X.columns[i],
            "vif": vif
        })

    return pd.DataFrame(vif_data)
