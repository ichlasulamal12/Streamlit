import pandas as pd
import numpy as np
from utils.io_handler import read_excel_file

def remove_duplicates(df):
    """
    Menghapus duplikat berdasarkan CSNO dengan memilih baris
    yang memiliki Date of Final PD paling akhir.
    """
    df["Date of Final PD"] = pd.to_datetime(df["Date of Final PD"])
    
    # Tetap gunakan struktur sebelumnya, hanya sorting berdasarkan 1 kolom
    df_sorted = df.sort_values(
        by=["Date of Final PD"],
        ascending=[False]
    )
    
    df_dedup = df_sorted.groupby("CSNO (CIF-CORE)", as_index=False).first()
    
    return df_dedup

def categorize_final_pd(df, segment):
    """
    Mengelompokkan Final_PD_2 ke dalam 7 grup berdasarkan segmen.
    """

    def map_group_sme(pd_value):
        if pd_value <= 0.0089:
            return "1"
        elif pd_value <= 0.0126:
            return "2"
        elif pd_value <= 0.0174:
            return "3"
        elif pd_value <= 0.0233:
            return "4"
        elif pd_value <= 0.0312:
            return "5"
        elif pd_value <= 0.0410:
            return "6"
        else:
            return "7"

    def map_group_wholesale(pd_value):
        # Ganti sesuai kriteria Wholesale
        if pd_value <= 0.007:
            return "1"
        elif pd_value <= 0.01:
            return "2"
        elif pd_value <= 0.015:
            return "3"
        elif pd_value <= 0.022:
            return "4"
        elif pd_value <= 0.03:
            return "5"
        elif pd_value <= 0.04:
            return "6"
        else:
            return "7"

    def map_group_mortgage(pd_value):
        # Ganti sesuai kriteria Mortgage
        if pd_value <= 0.005:
            return "1"
        elif pd_value <= 0.01:
            return "2"
        elif pd_value <= 0.015:
            return "3"
        elif pd_value <= 0.02:
            return "4"
        elif pd_value <= 0.03:
            return "5"
        elif pd_value <= 0.04:
            return "6"
        else:
            return "7"

    # Pilih fungsi berdasarkan segmen
    if segment == "SME":
        df["pd_group"] = df["Final PD_2"].apply(map_group_sme)
    elif segment == "Wholesale":
        df["pd_group"] = df["Final PD_2"].apply(map_group_wholesale)
    elif segment == "Mortgage":
        df["pd_group"] = df["Final PD_2"].apply(map_group_mortgage)
    else:
        raise ValueError("Segmen tidak dikenali")

    return df

def add_expected(df):
    """
    Menambahkan nilai expected untuk setiap kelompok pd_group.
    """
    expected_dict = {
        "1": 0.0666,
        "2": 0.1169,
        "3": 0.2512,
        "4": 0.2397,
        "5": 0.1793,
        "6": 0.0826,
        "7": 0.0636
    }
    df["expected"] = df["pd_group"].map(expected_dict)
    return df

def preprocess_for_psi(file_path, segment):
    """
    Pipeline lengkap: baca file → bersihkan → kelompokkan → tambah expected.
    """
    df = read_excel_file(file_path)
    df = remove_duplicates(df)
    df = categorize_final_pd(df, segment)
    df = add_expected(df)
    return df

def calculate_psi(df):
    """
    Menghitung nilai PSI berdasarkan kelompok 1 hingga 7 dari kolom Final_PD_2.
    Diasumsikan dataframe sudah melalui proses:
    - duplikat dihapus
    - pengelompokan Final_PD_2 ke kelompok 1–7
    - penambahan nilai expected untuk setiap kelompok
    
    Kolom yang digunakan:
    - 'group' → hasil pengelompokan
    - 'expected' → nilai expected dari masing-masing group
    """

    # Hitung count dan actual distribution
    group_count = df["pd_group"].value_counts().sort_index()
    total = group_count.sum()
    actual_pct = group_count / total

    # Buat dataframe gabungan
    psi_df = pd.DataFrame({
        "pd_group": group_count.index,
        "total":group_count.values,
        "actual_pct": actual_pct.values,
        "expected": [df[df["pd_group"] == g]["expected"].iloc[0] for g in group_count.index]
    })

    # Hitung log(actual/expected), jika error (misal 0/0), jadikan 0
    def safe_log_ratio(actual, expected):
        try:
            return np.log(actual / expected)
        except:
            return 0

    psi_df["log_ratio"] = psi_df.apply(
        lambda row: safe_log_ratio(row["actual_pct"], row["expected"]),
        axis=1
    )

    # Hitung indeks per grup
    psi_df["index"] = (psi_df["actual_pct"] - psi_df["expected"]) * psi_df["log_ratio"]

    # Jumlahkan untuk mendapatkan nilai PSI akhir
    psi_value = psi_df["index"].sum()

    return psi_value, psi_df
