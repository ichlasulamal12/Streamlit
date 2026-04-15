import pandas as pd
import numpy as np

def calculate_woe_iv(df, feature, target, alpha=0.5):

    grouped = df.groupby(feature)[target].agg(["count", "sum"]).reset_index()
    grouped.columns = ["bin", "total", "bad"]

    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    n_bins = len(grouped)

    # 🔥 SMOOTHING
    grouped["good_dist"] = (grouped["good"] + alpha) / (total_good + alpha * n_bins)
    grouped["bad_dist"] = (grouped["bad"] + alpha) / (total_bad + alpha * n_bins)

    # WOE
    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])

    # IV
    grouped["iv_contrib"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    iv = grouped["iv_contrib"].sum()

    return grouped, iv
