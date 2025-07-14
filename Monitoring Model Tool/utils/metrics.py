import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from utils.io_handler import read_excel_file

# Fungsi bantu untuk ambil tahun & bulan dari format "YYYY.MM"
def extract_year_month(period: str):
    year, month = map(int, period.split("."))
    return year, month

# Membaca semua file CSV DPD menjadi dictionary
def load_search_dpd(dpd_months: list[str], dpd_dir: str):
    search_dpd = {}
    for month in dpd_months:
        file_name = os.path.join(dpd_dir, f"search_dpd_{month}.csv")
        if os.path.exists(file_name):
            try:
                df = pd.read_csv(file_name, usecols=["zacno", "dpd"])
            except:
                df = pd.DataFrame(columns=["zacno", "dpd"])
        else:
            df = pd.DataFrame(columns=["zacno", "dpd"])
        search_dpd[month] = df
    return search_dpd

# Fungsi untuk menghasilkan list bulan dalam format MMYY
def generate_months(start_year: int, start_month: int, num_periods: int):
    months = []
    for i in range(num_periods):
        date = datetime(start_year, start_month, 1) + relativedelta(months=i)
        months.append(date.strftime("%m%y"))
    return months

# Fungsi utama proses per periode
def process_max_dpd_per_observation(input_file: str, dpd_dir: str, sheet_names: list[str], dpd_months: list[str]):
    search_dpd_dict = load_search_dpd(dpd_months, dpd_dir)

    writer = pd.ExcelWriter("output/max_dpd_flag_output v2.xlsx", engine='openpyxl')
    all_combined = []

    start_year, start_month = extract_year_month(sheet_names[0])
    end_year, end_month = extract_year_month(sheet_names[-1])

    month_lists = {}
    for period in sheet_names:
        y, m = extract_year_month(period)
        next_month_date = datetime(y, m, 1) + relativedelta(months=1)
        month_lists[period] = generate_months(next_month_date.year, next_month_date.month, 12)

    data = read_excel_file(input_file)
    # Ubah kolom tanggal
    data["Open Date"] = pd.to_datetime(data["Open Date"])
    data["YYYY_MM"] = data["Open Date"].dt.strftime("%Y.%m")

    # Kelompokkan berdasarkan YYYY.MM
    grouped = dict(tuple(data.groupby("YYYY_MM")))

    # Tambahkan 12 bulan ke depan dan kolom Max_DPD, Bad_Flag
    data_dict = {}

    for key, df in grouped.items():
        df = df.copy()

        # Tambah kolom 12 bulan ke depan
        base_date = datetime.strptime(key, "%Y.%m")
        for i in range(1, 13):
            next_month = (base_date + relativedelta(months=i)).strftime("%Y.%m")
            df[next_month] = pd.NA

        # Tambah kolom Max_DPD dan Bad_Flag di akhir
        df["Max_DPD"] = pd.NA
        df["Bad_Flag"] = pd.NA

        # Simpan ke dictionary
        data_dict[key] = df

    for period in sheet_names:
        df = data_dict[period].copy()
        df = df.iloc[:, :19]
        df.rename(columns={"ACNO": "zacno"}, inplace=True)
    
        for mon in month_lists[period]:
            mon_full = f"20{mon[2:]}" + "." + mon[:2]
            temp = search_dpd_dict.get(mon, pd.DataFrame(columns=["zacno", "dpd"]))
            df = df.merge(temp, on="zacno", how="left")
            df.rename(columns={"dpd": mon_full}, inplace=True)

        dpd_cols = df.columns[20:]
        df["Max DPD"] = df[dpd_cols].max(axis=1, skipna=True)
        df["Bad Flag"] = (df["Max DPD"] > 90).astype(int)
        df.insert(0, "Sheet", period)

        df.to_excel(writer, sheet_name=period, index=False)
    
        # 🔁 Rename kolom YYYY.MM → M1~M12 untuk versi All
        df_all = df.copy()
        date_cols = [col for col in df_all.columns if col[:4].isdigit() and "." in col]
        rename_map = {old: f"M{i+1}" for i, old in enumerate(sorted(date_cols))}
        df_all.rename(columns=rename_map, inplace=True)
        all_combined.append(df_all)

    df_all = pd.concat(all_combined, ignore_index=True)
    df_all.to_excel(writer, sheet_name="All", index=False)
    writer.close()

    return df_all

# Fungsi untuk menghapus duplikat berdasarkan CSNO dengan aturan berlapis
def deduplicate_gini(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = (
        df.sort_values([
            "CSNO (CIF-CORE)",
            "Bad Flag",
            "Max DPD",
            "Open Date",
            "Date of Final PD"
        ], ascending=[True, False, False, False, False])
        .drop_duplicates(subset=["CSNO (CIF-CORE)"], keep="first")
    )
    return df_sorted

# Fungsi untuk menghitung Gini, KS, dan AUROC berdasarkan segment
def calculate_gini_metrics(df: pd.DataFrame, segment: str, score_col: str = "Final PD", flag_col: str = "Bad Flag"):
    if segment == "SME":
        bins = [0, 0.0089, 0.0126, 0.0174, 0.0233, 0.0312, 0.0410, 1]
        labels = list(range(1, 8))
    elif segment == "Wholesale":
        bins = [0, 0.007, 0.010, 0.014, 0.020, 0.028, 0.037, 1]  # contoh bin, sesuaikan bila perlu
        labels = list(range(1, 8))
    elif segment == "Mortgage":
        bins = [0, 0.005, 0.009, 0.013, 0.018, 0.024, 0.031, 1]  # contoh bin, sesuaikan bila perlu
        labels = list(range(1, 8))
    else:
        raise ValueError("Segment tidak dikenali. Harus SME, Wholesale, atau Mortgage.")

    df = df.copy()
    df = df[df[score_col].notnull()].copy()
    df["Group"] = pd.cut(df[score_col], bins=bins, labels=labels, include_lowest=True)

    grouped = df.groupby("Group", observed=False)
    result = grouped.agg(
        bad=(flag_col, lambda x: (x == 1).sum()),
        good=(flag_col, lambda x: (x == 0).sum()),
        total=(flag_col, 'count')
    ).reset_index()

    total_bad = result["bad"].sum()
    total_good = result["good"].sum()
    total_total = result["total"].sum()

    result["bad_rate"] = result["bad"] / result["total"]
    result["prop_bad"] = result["bad"] / total_bad
    result["prop_good"] = result["good"] / total_good
    result["prop_total"] = result["total"] / total_total

    result = result.sort_values("Group", ascending=False).reset_index(drop=True)
    result["cum_bad"] = result["prop_bad"].cumsum()
    result["cum_good"] = result["prop_good"].cumsum()
    result["cum_total"] = result["prop_total"].cumsum()

    result["ks"] = abs(result["cum_good"].shift(-1).fillna(0) - result["cum_bad"].shift(-1).fillna(0))
    result["roc"] = 0.5 * result["prop_good"] * result["prop_bad"] + (1 - result["cum_good"]) * result["prop_bad"]

    ks_value = result["ks"].max()
    auroc_value = result["roc"].sum()
    gini_value = (auroc_value * 2) - 1

    return df, result, ks_value, auroc_value, gini_value
