import streamlit as st
import pandas as pd

from database.models import create_tables
from database.crud import get_projects, create_project, delete_project

# ======================
# INIT
# ======================
st.set_page_config(page_title="Credit Risk App", layout="wide")
create_tables()

st.markdown("""
<style>
/* Card-like container */
.block-container {
    padding-top: 2rem;
}

/* Data editor styling */
div[data-testid="stDataEditor"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E5E7E9;
}

/* Button styling */
button[kind="primary"] {
    border-radius: 8px;
    font-weight: 600;
}

/* General button */
button {
    border-radius: 8px !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #F4F6F7;
}

section[data-testid="stSidebar"] button {
    border-radius: 8px;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📊 Credit Risk App")

    # Navigation
    if "page" not in st.session_state:
        st.session_state["page"] = "project_list"

    page = st.radio(
        "Navigation",
        ["Project List", "Project Dashboard"],
        index=0 if st.session_state["page"] == "project_list" else 1
    )

    st.session_state["page"] = (
        "project_list" if page == "Project List" else "project_dashboard"
    )

    # Active project
    project_name = st.session_state.get("project_name")
    if project_name:
        st.markdown(f"""
        <div style="padding:10px;border-radius:8px;
        background-color:#EAF2F8;">
        <b>Active Project</b><br>
        {project_name}
        </div>
        """, unsafe_allow_html=True)

    # Pipeline (only if dashboard)
    if st.session_state["page"] == "project_dashboard":
        st.markdown("### 🧭 Pipeline")

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

        if not st.session_state.get("project_id"):
            st.warning("Select a project first")
        else:
            for label, key in modules:
                if st.button(label, key=f"module_{key}", width='stretch'):
                    st.session_state["active_module"] = key


# ======================
# DIALOG: CREATE PROJECT
# ======================
@st.dialog("Create New Project")
def create_project_dialog():
    st.info("Enter a project name to create a new credit risk modelling workspace")
    project_name = st.text_input("Project Name")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("OK", key="create_ok"):
            if project_name.strip() == "":
                st.warning("Project name cannot be empty")
                return

            create_project(project_name)
            st.success("Project created!")
            st.rerun()

    with col2:
        if st.button("Cancel", key="create_cancel"):
            st.rerun()

@st.dialog("⚠️ Confirm Delete")
def delete_project_dialog(project_id, project_name):

    st.warning(f"Are you sure you want to delete project:")
    st.caption("This action cannot be undone")
    st.write(f"**{project_name}**")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑 Confirm Delete", key=f"delete_confirm_{project_id}"):
            delete_project(project_id)
            st.success("Project deleted")
            st.rerun()

    with col2:
        if st.button("Cancel", key="delete_cancel"):
            st.rerun()


# ======================
# PAGE 1: PROJECT LIST
# ======================
def project_list_page():
    st.markdown("""
    <div style="padding:20px;border-radius:12px;
    background: linear-gradient(90deg,#2E86C1,#5DADE2);
    color:white;">
        <h2>📊 Credit Risk Modelling</h2>
        <p>Manage and monitor credit risk projects</p>
    </div>
    """, unsafe_allow_html=True)

    projects = get_projects()
    df = pd.DataFrame(projects)

    col_left, col_right = st.columns([1, 3])

    # ======================
    # RIGHT: PROJECT TABLE
    # ======================
    with col_right:
        st.subheader("📁 Project List")

        if not df.empty:
            st.metric("Total Projects", len(df))
            df_display = df[["name", "created_at"]].copy()
            df_display["created_at"] = pd.to_datetime(df_display["created_at"]).dt.strftime("%Y-%m-%d")
            df_display.insert(0, "Select", False)

            edited_df = st.data_editor(
                df_display,
                width='stretch',
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

                if selected_project_name:
                    st.markdown(f"""
                    <div style="padding:10px;border-radius:8px;
                    background-color:#E8F8F5;">
                    ✅ <b>Selected Project:</b> {selected_project_name}
                    </div>
                    """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style="background-color:#ffffff;padding:15px;border-radius:12px;
        box-shadow:0 2px 6px rgba(0,0,0,0.1);">
        <h4>⚙️ Actions</h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # ======================
        # CREATE PROJECT (POPUP)
        # ======================
        if st.button("➕ Create Project", width='stretch'):
            create_project_dialog()

        # ======================
        # OPEN PROJECT
        # ======================
        if st.button("📂 Open Project", type="primary", width='stretch'):
            if selected_project_id is not None:
                st.session_state["project_id"] = selected_project_id
                st.session_state["project_name"] = selected_project_name
                st.session_state["page"] = "project_dashboard"
                st.session_state["active_module"] = None
                st.rerun()
            else:
                st.warning("Select a project first")

        # ======================
        # DELETE PROJECT
        # ======================
        if st.button("🗑 Delete Project", width='stretch'):
            if selected_project_id is not None:
                delete_project_dialog(selected_project_id, selected_project_name)
            else:
                st.warning("Select a project first")

# ======================
# PAGE 2: PROJECT DASHBOARD
# ======================
def project_dashboard():    
    project_id = st.session_state.get("project_id")
    project_name = st.session_state.get("project_name")

    if project_id is None:
        st.warning("No active project. Please select a project from Project List")
        return    

    st.title("📁 Project Dashboard")

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

    if project_name:
        st.markdown(f"""
        <div style="background-color:#EAF2F8;padding:20px;border-radius:12px;">
        <h3>📁 {project_name}</h3>
        <p>Active Project</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Project name not found")

    # ======================
    # BACK BUTTON
    # ======================
    col1, col2 = st.columns([8,1])

    with col2:    
        if st.button("⬅ Back"):
            st.session_state["page"] = "project_list"
            st.session_state["active_module"] = None
            st.rerun()

    st.divider()

    # ======================
    # LOAD MODULE
    # ======================
    active = st.session_state.get("active_module")

    if active:
        st.markdown(f"""
        <div style="padding:8px;border-radius:8px;
        background-color:#D6EAF8;">
        Active: <b>{active.upper()}</b>
        </div>
        """, unsafe_allow_html=True)

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
