import streamlit as st
import pandas as pd

from database.models import create_tables
from database.crud import get_projects, create_project, delete_project

# ======================
# INIT
# ======================
create_tables()
st.set_page_config(page_title="Credit Risk App", layout="wide")

# ======================
# DIALOG: CREATE PROJECT
# ======================
@st.dialog("Create New Project")
def create_project_dialog():

    project_name = st.text_input("Project Name")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("OK"):
            if project_name.strip() == "":
                st.warning("Project name cannot be empty")
                return

            create_project(project_name)
            st.success("Project created!")
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.rerun()

@st.dialog("⚠️ Confirm Delete")
def delete_project_dialog(project_id, project_name):

    st.warning(f"Are you sure you want to delete project:")
    st.error("This action cannot be undone")
    st.write(f"**{project_name}**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑 Confirm Delete"):
            delete_project(project_id)
            st.success("Project deleted")
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.rerun()

# ======================
# STATE NAVIGATION
# ======================
if "page" not in st.session_state:
    st.session_state["page"] = "project_list"

# ======================
# PAGE 1: PROJECT LIST
# ======================
def project_list_page():
    st.title("📊 Credit Risk Modelling")

    projects = get_projects()
    df = pd.DataFrame(projects)

    col_left, col_right = st.columns([1, 3])

    # ======================
    # RIGHT: PROJECT TABLE
    # ======================
    with col_right:
        st.subheader("📁 Project List")

        if not df.empty:
            df_display = df[["name", "created_at"]].copy()
            df_display.insert(0, "Select", False)

            edited_df = st.data_editor(
                df_display,
                use_container_width=True,
                hide_index=True,
                key="project_table",
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", width="small"),
                    "name": st.column_config.TextColumn("Project Name"),
                    "created_at": st.column_config.TextColumn("Created At"),
                }
            )

            selected_rows = edited_df[edited_df["Select"] == True]

            if len(selected_rows) > 1:
                st.warning("Select only one project")

            if not selected_rows.empty:
                selected_index = selected_rows.index[0]
                selected_project_id = int(df.loc[selected_index, "id"])
                selected_project_name = df.loc[selected_index, "name"]
            else:
                selected_project_id = None
                selected_project_name = None

        else:
            st.info("No project available")
            selected_project_id = None
            selected_project_name = None

    # ======================
    # LEFT: ACTION PANEL
    # ======================
    with col_left:
        st.subheader("⚙️ Actions")

        # ======================
        # CREATE PROJECT (POPUP)
        # ======================
        if st.button("➕ Create Project", use_container_width=True):
            create_project_dialog()

        # ======================
        # OPEN PROJECT
        # ======================
        if st.button("📂 Open Project", use_container_width=True):
            if selected_project_id:
                st.session_state["project_id"] = selected_project_id
                st.session_state["project_name"] = selected_project_name
                st.session_state["page"] = "project_dashboard"
                st.rerun()
            else:
                st.warning("Select a project first")

        # ======================
        # DELETE PROJECT
        # ======================
        if st.button("🗑 Delete Project", use_container_width=True):
            if selected_project_id:
                delete_project_dialog(selected_project_id, selected_project_name)
            else:
                st.warning("Select a project first")

# ======================
# PAGE 2: PROJECT DASHBOARD
# ======================
def project_dashboard():
    project_id = st.session_state.get("project_id")
    project_name = st.session_state.get("project_name")

    st.title("📁 Project Dashboard")

    if project_name:
        st.success(f"Active Project: {project_name}")
    else:
        st.warning("Project name not found")

    # ======================
    # BACK BUTTON
    # ======================
    if st.button("⬅ Back to Project List"):
        st.session_state["page"] = "project_list"
        st.rerun()

    st.divider()

    # ======================
    # MODULE NAVIGATION
    # ======================
    st.subheader("🧭 Modelling Pipeline")
    st.markdown("➡️ Proceed step by step from ① to ⑨")

    modules = [
        ("① Input Data", "input"),
        ("② Preprocessing", "preprocessing"),
        ("③ Split Data", "split"),
        ("④ Binning", "binning"),
        ("⑤ WOE", "woe"),
        ("⑥ Multicollinearity", "vif"),
        ("⑦ SMOTE", "smote"),
        ("⑧ Training Model", "training"),
        ("⑨ Model Performance", "performance"),
    ]

    cols = st.columns(3)

    for i, (label, key) in enumerate(modules):
        col = cols[i % 3]

        if col.button(label, use_container_width=True):
            st.session_state["active_module"] = key

    # ======================
    # LOAD MODULE
    # ======================
    active = st.session_state.get("active_module")
    if active:
        st.info(f"Active Module: {active.upper()}")

    st.divider()

    if active == "input":
        from modules import input_data
        input_data.run(project_id)

    elif active == "preprocessing":
        from modules import preprocessing
        preprocessing.run(project_id)

    elif active == "split":
        from modules import split_data
        split_data.run(project_id)

    elif active == "binning":
        from modules import binning
        binning.run(project_id)

    elif active == "woe":
        from modules import woe
        woe.run(project_id)

    elif active == "vif":
        from modules import multicollinearity
        multicollinearity.run(project_id)

    elif active == "smote":
        from modules import smote
        smote.run(project_id)

    elif active == "training":
        from modules import training
        training.run(project_id)

    elif active == "performance":
        from modules import model_performance
        model_performance.run(project_id)

# ======================
# ROUTER
# ======================
if st.session_state["page"] == "project_list":
    project_list_page()
else:
    project_dashboard()
