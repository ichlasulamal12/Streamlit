import numpy as np
import pandas as pd


def apply_transformation(df, binning_rules):
    df_out = df.copy()

    for col, rule in binning_rules.items():

        if rule.get("type") != "numeric":
            continue

        transform = rule.get("transform", "none")

        if transform == "none":
            continue

        if transform["type"] == "log1p":
            shift = transform.get("shift", 0)

            if shift > 0:
                df_out[col] = np.log1p(df_out[col] + shift)
            else:
                df_out[col] = np.log1p(df_out[col])

    return df_out
