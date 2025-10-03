import pandas as pd
import numpy as np
from utils.io_handler import read_excel_file

def remove_duplicates(df, segment):
    """
    Menghapus duplikat berdasarkan kolom ID dan tanggal sesuai segmen.
    Untuk SME: gunakan 'CSNO (CIF-CORE)' dan 'Date of Final PD'.
    Untuk Wholesale: gunakan 'CIF M18' dan 'Tanggal Proses Rating'.
    """
    if segment == "SME":
        id_col = "CSNO (CIF-CORE)"
        date_col = "Date of Final PD"
    elif segment == "Wholesale":
        id_col = "CIF M18"
        date_col = "Tanggal Proses Rating"
    else:
        raise ValueError(f"Segment '{segment}' tidak dikenali.")

    df[date_col] = pd.to_datetime(df[date_col])
    
    df_sorted = df.sort_values(by=[date_col], ascending=False)
    df_dedup = df_sorted.groupby(id_col, as_index=False).first()
    
    return df_dedup

def categorize_final_pd(df, segment):
    """
    Mengelompokkan Final_PD_2 ke dalam beberapa grup berdasarkan segmen.
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
        if pd_value == "Grade 1":
            return "1"
        elif pd_value == "Grade 2":
            return "2"
        elif pd_value == "Grade 3":
            return "3"
        elif pd_value == "Grade 4":
            return "4"
        elif pd_value == "Grade 5":
            return "5"
        elif pd_value == "Grade 6":
            return "6"
        elif pd_value == "Grade 7":
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

def add_expected(df, segment):
    """
    Menambahkan nilai expected untuk setiap kelompok pd_group sesuai segmen.
    Untuk Wholesale, dipisah berdasarkan size: large (1) dan medium (2).
    """
    expected_dicts = {
        "SME": {
            "1": 0.0666,
            "2": 0.1169,
            "3": 0.2512,
            "4": 0.2397,
            "5": 0.1793,
            "6": 0.0826,
            "7": 0.0636
        },
        "Wholesale_Large": {
            "1": 0.0392,
            "2": 0.3115,
            "3": 0.3530,
            "4": 0.1468,
            "5": 0.0953,
            "6": 0.0433,
            "7": 0.0108
        },
        "Wholesale_Medium": {
            "1": 0.0121,
            "2": 0.1632,
            "3": 0.3002,
            "4": 0.1848,
            "5": 0.1675,
            "6": 0.1160,
            "7": 0.0561
        },
        "Mortgage": {
            "1": 0.0011,
            "2": 0.0020,
            "3": 0.0051,
            "4": 0.0205,
            "5": 0.0614,
            "6": 0.2023,
            "7": 0.7076
        }
    }

    if segment == "Wholesale":
        # Untuk wholesale, kita perlu memisahkan berdasarkan kolom 'Size'
        # Kolom 'Size' berisi 1 untuk large dan 2 untuk medium
        if 'Size' not in df.columns:
            raise ValueError("Kolom 'Size' tidak ditemukan dalam dataset. Pastikan dataset memiliki kolom 'Size' dengan nilai 1 (large) atau 2 (medium).")
        
        # Buat kolom expected berdasarkan Size
        df["expected"] = df.apply(lambda row: 
            expected_dicts["Wholesale_Large"][row["pd_group"]] if row["Size"] == 1 
            else expected_dicts["Wholesale_Medium"][row["pd_group"]], axis=1)
    else:
        expected_dict = expected_dicts.get(segment)
        if expected_dict is None:
            raise ValueError(f"Segment '{segment}' tidak dikenali.")
        df["expected"] = df["pd_group"].astype(str).map(expected_dict)
    return df

def preprocess_for_psi(file_path, segment):
    """
    Pipeline lengkap: baca file → bersihkan → kelompokkan → tambah expected.
    """
    df = read_excel_file(file_path)
    df = remove_duplicates(df, segment)
    df = categorize_final_pd(df, segment)
    df = add_expected(df, segment)
    return df

def calculate_psi(df, segment):
    """
    Menghitung nilai PSI berdasarkan kelompok 1 hingga 7 dari kolom Final_PD_2.
    Untuk Wholesale, menghitung PSI terpisah untuk setiap size (large=1, medium=2).
    
    Diasumsikan dataframe sudah melalui proses:
    - duplikat dihapus
    - pengelompokan Final_PD_2 ke kelompok 1–7
    - penambahan nilai expected untuk setiap kelompok
    
    Kolom yang digunakan:
    - 'pd_group' → hasil pengelompokan
    - 'expected' → nilai expected dari masing-masing group
    - 'size' → untuk wholesale (1=large, 2=medium)
    """
    
    if segment == "Wholesale":
        # Untuk wholesale, hitung PSI terpisah untuk setiap size
        results = {}
        
        for size_value in [1, 2]:
            size_name = "Large" if size_value == 1 else "Medium"
            df_size = df[df['Size'] == size_value].copy()
            
            if len(df_size) == 0:
                # Jika tidak ada data untuk size ini, skip
                continue
                
            # Hitung count dan actual distribution untuk size ini
            group_count = df_size["pd_group"].value_counts().sort_index()
            total = group_count.sum()
            actual_pct = group_count / total

            # Buat dataframe gabungan untuk size ini
            psi_df = pd.DataFrame({
                "pd_group": group_count.index,
                "total": group_count.values,
                "actual_pct": actual_pct.values,
                "expected": [df_size[df_size["pd_group"] == g]["expected"].iloc[0] for g in group_count.index]
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
            
            # Tambahkan kolom Size untuk identifikasi
            psi_df["Size"] = size_value
            psi_df["size_name"] = size_name
            
            results[f"Wholesale_{size_name}"] = {
                "psi_value": psi_value,
                "psi_df": psi_df
            }
        
        return results
    else:
        # Untuk segmen lain (SME, Mortgage), hitung PSI seperti biasa
        # Hitung count dan actual distribution
        group_count = df["pd_group"].value_counts().sort_index()
        total = group_count.sum()
        actual_pct = group_count / total

        # Buat dataframe gabungan
        psi_df = pd.DataFrame({
            "pd_group": group_count.index,
            "total": group_count.values,
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