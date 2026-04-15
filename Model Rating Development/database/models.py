from database.db import get_connection


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # ======================
    # PROJECTS
    # ======================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ======================
    # DATASETS
    # ======================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        project_id INTEGER PRIMARY KEY,
        file_name TEXT,
        data BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ======================
    # PREPROCESSING
    # ======================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preprocessing (
        project_id INTEGER PRIMARY KEY,
        target TEXT,
        features TEXT,
        imputation_rules TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ======================
    # DATA SPLIT
    # ======================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS data_split (
        project_id INTEGER PRIMARY KEY,
        train_data BLOB,
        test_data BLOB,
        val_data BLOB,
        method TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ======================
    # BINNING
    # ======================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS binning (
        project_id INTEGER PRIMARY KEY,
        binning_rules TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ======================
    # MODEL DATASET (WOE)
    # ======================
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_dataset (
        project_id INTEGER PRIMARY KEY,
        df_woe BLOB,
        features TEXT,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ======================
    # COMMIT & CLOSE
    # ======================
    conn.commit()
    conn.close()
