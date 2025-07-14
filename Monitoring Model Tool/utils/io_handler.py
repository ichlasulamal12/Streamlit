import pandas as pd

def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df

def save_to_excel(df, output_path):
    df.to_excel(output_path, index=False)
