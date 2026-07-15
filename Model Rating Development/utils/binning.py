import pandas as pd
import numpy as np
from optbinning import OptimalBinning


# =====================================================
# QUANTILE NUMERIC BINNING
# =====================================================
def create_numeric_bins(series, n_bins=5, separate_missing=False):

    series_clean = pd.to_numeric(series, errors="coerce")

    if not separate_missing:
        return pd.qcut(
            series_clean,
            q=n_bins,
            duplicates="drop"
        )

    non_missing = series_clean.dropna()

    bins = pd.qcut(
        non_missing,
        q=n_bins,
        duplicates="drop"
    )

    result = pd.Series(index=series.index, dtype=object)

    result.loc[non_missing.index] = bins.astype(str)
    result.loc[series_clean.isna()] = "Missing"

    return result


# =====================================================
# CATEGORICAL BINNING
# =====================================================
def create_categorical_bins(series, separate_missing=False):

    if separate_missing:
        return (
            series
            .fillna("Missing")
            .astype(str)
        )

    return series.astype(str)


# =====================================================
# MANUAL NUMERIC BINNING
# =====================================================
def create_manual_numeric_bins(
    series,
    cut_points,
    separate_missing=False
):

    series_clean = pd.to_numeric(series, errors="coerce")

    cut = pd.cut(
        series_clean,
        bins=[-float("inf")] + sorted(cut_points) + [float("inf")]
    )

    if not separate_missing:
        return cut

    result = cut.astype(str)
    result[series_clean.isna()] = "Missing"

    return result


# =====================================================
# MANUAL CATEGORICAL BINNING
# =====================================================
def create_manual_categorical_bins(
    series,
    mapping_dict,
    separate_missing=False
):

    result = series.map(mapping_dict)

    if separate_missing:
        result = result.fillna("Missing")
    else:
        result = result.fillna("Other")

    return result.astype(str)


# =====================================================
# OPTIMAL BINNING
# =====================================================
def create_optimal_bins(
    series,
    target,
    monotonic_trend="auto",
    separate_missing=False
):

    series_clean = pd.to_numeric(series, errors="coerce")

    if separate_missing:

        non_missing = series_clean.notna()

        x = series_clean[non_missing]
        y = target[non_missing]

    else:

        x = series_clean
        y = target

    optb = OptimalBinning(
        dtype="numerical",
        solver="mip",
        monotonic_trend=monotonic_trend,
        max_n_prebins=20
    )

    optb.fit(x, y)

    bins = optb.transform(x, metric="bins")

    if not separate_missing:
        return bins, optb

    result = pd.Series(index=series.index, dtype=object)

    result.loc[x.index] = bins.astype(str)
    result.loc[series_clean.isna()] = "Missing"

    return result, optb


# =====================================================
# BIN STATISTICS
# =====================================================
def calculate_bin_stats(
    df,
    feature,
    target,
    bins,
    separate_missing=False
):

    temp = pd.DataFrame({
        "feature": bins,
        "target": df[target]
    })

    if separate_missing:
        temp["feature"] = (
            temp["feature"]
            .astype(object)
            .fillna("Missing")
        )

    grouped = temp.groupby(
        "feature",
        dropna=False,
        observed=False
    )

    result = grouped.agg(
        total=("target", "count"),
        bad=("target", "sum")
    ).reset_index()

    result["good"] = result["total"] - result["bad"]
    result["bad_ratio"] = result["bad"] / result["total"]
    result["portion"] = result["total"] / result["total"].sum()

    # Missing di paling atas
    if separate_missing:
        if "Missing" in result["feature"].astype(str).values:

            missing = result[
                result["feature"].astype(str) == "Missing"
            ]

            other = result[
                result["feature"].astype(str) != "Missing"
            ]

            result = pd.concat(
                [missing, other],
                ignore_index=True
            )

    return result


# =====================================================
# APPLY BINNING
# =====================================================
def apply_binning(df, rules):

    df_copy = df.copy()

    for col, rule in rules.items():

        separate_missing = rule.get(
            "separate_missing",
            False
        )

        # ============================================
        # NUMERIC
        # ============================================
        if rule["type"] == "numeric":

            series = pd.to_numeric(
                df_copy[col],
                errors="coerce"
            )

            if rule["mode"] == "quantile":

                result = pd.qcut(
                    series,
                    q=rule["n_bins"],
                    duplicates="drop"
                )

            elif rule["mode"] == "optimal":

                bins = (
                    [-float("inf")]
                    + rule["splits"]
                    + [float("inf")]
                )

                result = pd.cut(series, bins=bins)

            else:

                bins = (
                    [-float("inf")]
                    + rule["cut_points"]
                    + [float("inf")]
                )

                result = pd.cut(series, bins=bins)

            if separate_missing:
                result = result.astype(str)
                result[series.isna()] = "Missing"

            df_copy[col] = result

        # ============================================
        # CATEGORICAL
        # ============================================
        else:

            if rule["mode"] == "quantile":

                if separate_missing:

                    df_copy[col] = (
                        df_copy[col]
                        .fillna("Missing")
                        .astype(str)
                    )

                else:

                    df_copy[col] = (
                        df_copy[col]
                        .astype(str)
                    )

            else:

                result = df_copy[col].map(
                    rule["mapping"]
                )

                if separate_missing:
                    result = result.fillna("Missing")
                else:
                    result = result.fillna("Other")

                df_copy[col] = result.astype(str)

    return df_copy
