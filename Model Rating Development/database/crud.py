import pickle
import json
from database.db import get_connection


# ======================
# CREATE PROJECT
# ======================
def create_project(name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO projects (name) VALUES (?)",
        (name,)
    )

    conn.commit()
    conn.close()


# ======================
# READ PROJECTS
# ======================
def get_projects():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM projects ORDER BY created_at DESC")
    rows = cursor.fetchall()

    result = [dict(row) for row in rows]

    conn.close()
    return result


# ======================
# DELETE PROJECT
# ======================
def delete_project(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    conn.commit()
    conn.close()


# ======================
# DATASET
# ======================
def save_dataset(project_id, df, file_name):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO datasets (project_id, file_name, data)
    VALUES (?, ?, ?)
    """, (
        project_id,
        file_name,
        pickle.dumps(df)
    ))

    conn.commit()
    conn.close()


def load_dataset(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM datasets WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return pickle.loads(row["data"]), row["file_name"]
    return None, None


# ======================
# PREPROCESSING
# ======================
def save_preprocessing(project_id, target, features, imputation_rules):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO preprocessing (project_id, target, features, imputation_rules)
    VALUES (?, ?, ?, ?)
    """, (
        project_id,
        target,
        json.dumps(features),
        json.dumps(imputation_rules)
    ))

    conn.commit()
    conn.close()


def load_preprocessing(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM preprocessing WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return {
            "target": row["target"],
            "features": json.loads(row["features"]),
            "imputation_rules": json.loads(row["imputation_rules"])
        }

    return None


# ======================
# DATA SPLIT
# ======================
def save_split(project_id, train, test, val, method):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO data_split
    (project_id, train_data, test_data, val_data, method)
    VALUES (?, ?, ?, ?, ?)
    """, (
        project_id,
        pickle.dumps(train),
        pickle.dumps(test),
        pickle.dumps(val) if val is not None else None,
        method
    ))

    conn.commit()
    conn.close()


def load_split(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM data_split WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()

    conn.close()

    if row:
        return {
            "train": pickle.loads(row["train_data"]),
            "test": pickle.loads(row["test_data"]),
            "val": pickle.loads(row["val_data"]) if row["val_data"] else None,
            "method": row["method"]
        }

    return None


# ======================
# BINNING
# ======================
def save_binning(project_id, rules):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO binning (project_id, binning_rules)
    VALUES (?, ?)
    """, (project_id, json.dumps(rules)))

    conn.commit()
    conn.close()


def load_binning(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM binning WHERE project_id = ?", (project_id,))
    row = cursor.fetchone()

    conn.close()

    return json.loads(row["binning_rules"]) if row else None


# ======================
# MODEL DATASET (FIXED)
# ======================
def ensure_model_dataset_schema(cursor):

    cursor.execute("PRAGMA table_info(model_dataset)")
    columns = [row[1] for row in cursor.fetchall()]

    if "woe_result" not in columns:
        cursor.execute("ALTER TABLE model_dataset ADD COLUMN woe_result BLOB")

    if "coef_df" not in columns:
        cursor.execute("ALTER TABLE model_dataset ADD COLUMN coef_df BLOB")

    if "intercept" not in columns:
        cursor.execute("ALTER TABLE model_dataset ADD COLUMN intercept REAL")


def save_model_dataset(
    project_id,
    df_woe=None,
    features=None,
    woe_result=None,
    coef_df=None,
    intercept=None,
    woe_maps=None,
    source="original"
):
    conn = get_connection()
    cursor = conn.cursor()

    ensure_model_dataset_schema(cursor)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_dataset (
            project_id INTEGER PRIMARY KEY,
            df_woe BLOB,
            features TEXT,
            woe_result BLOB,
            coef_df BLOB,
            intercept REAL,
            woe_maps BLOB,
            source TEXT
        )
    """)

    # ======================
    # 🔥 CHECK EXISTING DATA
    # ======================
    cursor.execute("SELECT * FROM model_dataset WHERE project_id=?", (project_id,))
    row = cursor.fetchone()

    if row:
        # ambil existing
        existing_df_woe = pickle.loads(row["df_woe"]) if row["df_woe"] else None
        existing_features = json.loads(row["features"]) if row["features"] else None
        existing_woe_result = pickle.loads(row["woe_result"]) if row["woe_result"] else None
        existing_coef_df = pickle.loads(row["coef_df"]) if row["coef_df"] else None
        existing_intercept = row["intercept"]
        existing_woe_maps = pickle.loads(row["woe_maps"]) if row["woe_maps"] else None

    else:
        existing_df_woe = None
        existing_features = None
        existing_woe_result = None
        existing_coef_df = None
        existing_intercept = None
        existing_woe_maps = None

    # ======================
    # 🔥 MERGE DATA (INI KUNCI)
    # ======================
    final_df_woe = df_woe if df_woe is not None else existing_df_woe
    final_features = features if features is not None else existing_features
    final_woe_result = woe_result if woe_result is not None else existing_woe_result
    final_coef_df = coef_df if coef_df is not None else existing_coef_df
    final_intercept = intercept if intercept is not None else existing_intercept
    final_woe_maps = woe_maps if woe_maps is not None else existing_woe_maps

    # ======================
    # SAVE FINAL
    # ======================
    cursor.execute("""
        INSERT OR REPLACE INTO model_dataset
        (project_id, df_woe, features, woe_result, coef_df, intercept, woe_maps, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        project_id,
        pickle.dumps(final_df_woe) if final_df_woe is not None else None,
        json.dumps(final_features) if final_features is not None else None,
        pickle.dumps(final_woe_result) if final_woe_result is not None else None,
        pickle.dumps(final_coef_df) if final_coef_df is not None else None,
        final_intercept,
        pickle.dumps(final_woe_maps) if final_woe_maps is not None else None,
        source
    ))

    conn.commit()
    conn.close()


def load_model_dataset(project_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM model_dataset WHERE project_id=?
    """, (project_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "df_woe": pickle.loads(row["df_woe"]) if row["df_woe"] else None,
        "features": json.loads(row["features"]) if row["features"] else None,
        "woe_result": pickle.loads(row["woe_result"]) if row["woe_result"] else None,
        "coef_df": pickle.loads(row["coef_df"]) if row["coef_df"] else None,
        "intercept": row["intercept"],
        "source": row["source"]
    }


def save_model_rules(project_id, rating_rules=None, score_rules=None):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_rules (
            project_id TEXT PRIMARY KEY,
            rating_rules TEXT,
            score_rules TEXT
        )
    """)

    cursor.execute("""
        INSERT INTO model_rules (project_id, rating_rules, score_rules)
        VALUES (?, ?, ?)
        ON CONFLICT(project_id) DO UPDATE SET
            rating_rules=excluded.rating_rules,
            score_rules=excluded.score_rules
    """, (
        project_id,
        json.dumps(rating_rules) if rating_rules else None,
        json.dumps(score_rules) if score_rules else None
    ))

    conn.commit()
    conn.close()


def load_model_rules(project_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT rating_rules, score_rules
        FROM model_rules
        WHERE project_id = ?
    """, (project_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None, None

    rating_rules = json.loads(row[0]) if row[0] else []
    score_rules = json.loads(row[1]) if row[1] else []

    return rating_rules, score_rules
