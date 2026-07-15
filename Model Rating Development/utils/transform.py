import numpy as np
import pandas as pd


def apply_transformation(df, binning_rules):
    df_out = df.copy()

    for col, rule in binning_rules.items():

        if rule.get("type") != "numeric":
            continue

        transform = rule.get("transform")

        # ======================
        # SKIP JIKA TIDAK ADA TRANSFORM
        # ======================
        if not isinstance(transform, dict):
            continue

        if transform.get("type") == "log1p":

            shift = transform.get("shift", 0)

            # 🔥 HANDLE NEGATIVE VALUE
            if shift > 0:
                df_out[col] = np.log1p(df_out[col] + shift)
            else:
                # pastikan tidak log(negatif)
                series = pd.to_numeric(df_out[col], errors="coerce")

                series = series.clip(lower=0)

                df_out[col] = np.log1p(series)

    # ======================
    # CLEAN DATA
    # ======================
    df_out = df_out.replace([np.inf, -np.inf], np.nan)

    return df_out
