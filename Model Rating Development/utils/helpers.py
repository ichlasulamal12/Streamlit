import pandas as pd

def apply_imputation(df, imputation_rules):
    df_copy = df.copy()

    for col, rule in imputation_rules.items():

        if rule["method"] == "mean":
            value = df_copy[col].mean()

        elif rule["method"] == "median":
            value = df_copy[col].median()

        elif rule["method"] == "mode":
            value = df_copy[col].mode().iloc[0]

        elif rule["method"] == "manual":
            value = rule["value"]

        else:
            continue

        # 🔥 APPLY KE SEMUA MISSING
        df_copy[col] = df_copy[col].fillna(value)

    return df_copy
