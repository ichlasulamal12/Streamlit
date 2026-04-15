import pandas as pd
from optbinning import OptimalBinning

def create_numeric_bins(series, n_bins=5):

    # 🔥 convert ke numeric (critical fix)
    series_clean = pd.to_numeric(series, errors='coerce')

    return pd.qcut(series_clean, q=n_bins, duplicates='drop')


def create_categorical_bins(series):
    return series.astype(str)


def calculate_bin_stats(df, feature, target, bins):
    temp = pd.DataFrame({
        "feature": bins,
        "target": df[target]
    })

    grouped = temp.groupby("feature")

    result = grouped.agg(
        total=("target", "count"),
        bad=("target", "sum")
    ).reset_index()

    result["good"] = result["total"] - result["bad"]
    result["bad_ratio"] = result["bad"] / result["total"]
    result["portion"] = result["total"] / result["total"].sum()

    return result

def create_manual_numeric_bins(series, cut_points):

    series_clean = pd.to_numeric(series, errors='coerce')

    bins = [-float("inf")] + sorted(cut_points) + [float("inf")]

    return pd.cut(series_clean, bins=bins)

def create_manual_categorical_bins(series, mapping_dict):
    return series.map(mapping_dict).fillna("Other")

def apply_binning(df, rules):

    df_copy = df.copy()

    for col, rule in rules.items():

        # ======================
        # NUMERIC
        # ======================
        if rule["type"] == "numeric":

            series = pd.to_numeric(df_copy[col], errors='coerce')

            if rule["mode"] == "quantile":
                df_copy[col] = pd.qcut(
                    series,
                    q=rule["n_bins"],
                    duplicates='drop'
                )

            elif rule["mode"] == "optimal":
                bins = [-float("inf")] + rule["splits"] + [float("inf")]
                df_copy[col] = pd.cut(series, bins=bins)   

            else:
                bins = [-float("inf")] + rule["cut_points"] + [float("inf")]
                df_copy[col] = pd.cut(series, bins=bins)

        # ======================
        # CATEGORICAL
        # ======================
        else:

            if rule["mode"] == "quantile":
                df_copy[col] = df_copy[col].astype(str)

                # handle missing bin
                df_copy[col] = df_copy[col].fillna("MISSING")

            else:
                df_copy[col] = df_copy[col].map(rule["mapping"]).fillna("Other")

    return df_copy

def create_optimal_bins(series, target, monotonic_trend="auto"):

    series_clean = pd.to_numeric(series, errors='coerce')

    optb = OptimalBinning(
        dtype="numerical",
        solver="mip",              # 🔥 FIX
        monotonic_trend=monotonic_trend,
        max_n_prebins=20           # 🔥 FIX
    )

    optb.fit(series_clean, target)

    bins = optb.transform(series_clean, metric="bins")

    return bins, optb
