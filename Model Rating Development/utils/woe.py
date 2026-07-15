import pandas as pd
import numpy as np

def calculate_woe_iv(df, feature, target, alpha=0.05):

    df = df.copy()

    df[feature] = (
        df[feature]
        .fillna("Missing")
        .replace(["nan", "NaN", "None"], "Missing")
    )  
    grouped = (
        df.groupby(feature, dropna=False, observed=False)[target]
        .agg(["count", "sum"])
        .reset_index()
    )
    grouped.columns = ["bin", "total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    n_bins = len(grouped)

    # 🔥 SMOOTHING
    grouped["bad_rate"] = grouped["bad"] / grouped["total"]
    grouped["portion"] = grouped["total"] / grouped["total"].sum()
    grouped["good_dist"] = (grouped["good"] + alpha) / (total_good + alpha * n_bins)
    grouped["bad_dist"] = (grouped["bad"] + alpha) / (total_bad + alpha * n_bins)

    # WOE
    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])

    # IV
    grouped["iv_contrib"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    iv = grouped["iv_contrib"].sum()

    return grouped, iv

def sort_woe_table(df):

    def order(x):

        if str(x) == "Missing":
            return -999999999

        try:
            s = str(x)

            if "," in s:

                lower = (
                    s.split(",")[0]
                    .replace("(", "")
                    .replace("[", "")
                )

                if lower == "-inf":
                    return -1e30

                return float(lower)

        except:
            pass

        return 999999999

    df = df.copy()

    df["_sort"] = df["bin"].apply(order)

    df = df.sort_values("_sort").drop(columns="_sort")

    return df
