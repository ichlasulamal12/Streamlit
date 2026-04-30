import streamlit as st
import pandas as pd

from database.crud import save_dataset, load_dataset


def run(project_id):
    st.header("📥 Input Data")

    # ======================
    # LOAD EXISTING DATA
    # ======================
    df_existing, file_name = load_dataset(project_id)

    if df_existing is not None:
        st.success(f"Dataset loaded: {file_name}")

        st.write("### Preview Data")
        st.dataframe(df_existing.head(), width='stretch')

        st.write("### Dataset Info")
        st.write(f"Shape: {df_existing.shape}")

        if st.button("🔄 Replace Dataset"):
            st.session_state["replace_data"] = True

    else:
        st.info("No dataset uploaded yet")
        st.session_state["replace_data"] = True

    # ======================
    # UPLOAD SECTION
    # ======================
    if st.session_state.get("replace_data", False):

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx"]
        )

        if uploaded_file is not None:

            # 🔥 RESET SESSION (IMPORTANT)
            st.session_state.pop("converted_df", None)
            st.session_state.pop("type_config", None)

            try:
                # ======================
                # READ FILE
                # ======================
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success("File loaded successfully")

                # ======================
                # PREVIEW
                # ======================
                st.write("### Preview Data")
                st.dataframe(df.head(), width='stretch')

                st.write(f"Shape: {df.shape}")

                # ======================
                # TYPE ADJUSTMENT
                # ======================
                st.write("### 🔧 Column Type Adjustment")
                st.caption("⚠️ Conversion may introduce missing values")

                type_options = ["numeric", "categorical", "datetime", "string"]

                type_df = pd.DataFrame({
                    "Column": df.columns,
                    "Current Type": df.dtypes.astype(str)
                })

                # default mapping
                def map_default(dtype):
                    if "int" in dtype or "float" in dtype:
                        return "numeric"
                    elif "datetime" in dtype:
                        return "datetime"
                    elif "object" in dtype:
                        return "string"
                    else:
                        return "categorical"

                type_df["New Type"] = type_df["Current Type"].apply(map_default)

                edited_types = st.data_editor(
                    type_df,
                    width='stretch',
                    hide_index=True,
                    key="type_editor",
                    column_config={
                        "New Type": st.column_config.SelectboxColumn(
                            "New Type",
                            options=type_options
                        )
                    }
                )

                # simpan config
                st.session_state["type_config"] = edited_types

                # ======================
                # APPLY CONVERSION
                # ======================
                if st.button("⚙️ Apply Type Conversion"):

                    df_converted = df.copy()
                    edited_types = st.session_state["type_config"]

                    for _, row in edited_types.iterrows():
                        col = row["Column"]
                        new_type = row["New Type"]

                        try:
                            # NUMERIC
                            if new_type == "numeric":
                                df_converted[col] = (
                                    df_converted[col]
                                    .astype(str)
                                    .str.replace(",", "")
                                    .str.strip()
                                )
                                df_converted[col] = pd.to_numeric(
                                    df_converted[col],
                                    errors='coerce'
                                )

                            # DATETIME
                            elif new_type == "datetime":
                                df_converted[col] = pd.to_datetime(
                                    df_converted[col],
                                    errors='coerce',
                                    dayfirst=True
                                )

                            # CATEGORICAL
                            elif new_type == "categorical":
                                df_converted[col] = df_converted[col].astype("category")

                            # STRING
                            elif new_type == "string":
                                df_converted[col] = df_converted[col].astype(str)

                        except Exception as e:
                            st.error(f"Error converting {col}: {e}")

                    st.session_state["converted_df"] = df_converted

                    st.success("Conversion applied")

                # ======================
                # PREVIEW CONVERTED
                # ======================
                if "converted_df" in st.session_state:

                    df_conv = st.session_state["converted_df"]

                    st.write("### ✅ Converted Data Preview")
                    st.dataframe(df_conv.head(), width='stretch')

                    st.write("### Updated Types")
                    st.write(df_conv.dtypes)

                    # 🔥 missing impact
                    st.write("### Missing After Conversion")
                    st.write(df_conv.isna().sum())

                    # warning jika banyak missing
                    for col in df_conv.columns:
                        if df_conv[col].isna().mean() > 0.3:
                            st.warning(f"{col} has high missing after conversion")

                # ======================
                # SAVE BUTTON
                # ======================
                if st.button("💾 Save Dataset"):

                    final_df = st.session_state.get("converted_df", df)

                    save_dataset(project_id, final_df, uploaded_file.name)

                    st.success("Dataset saved with updated types!")

            except Exception as e:
                st.error(f"Error reading file: {e}")
