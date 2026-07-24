# ==========================================================
# Aurora PG College Students Placement Prediction System
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

FEATURES = [
    "CGPA",
    "Coding_Score",
    "Aptitude_Score",
    "Communication_Skills",
    "Projects",
    "Certifications",
    "Internship",
    "Attendance",
    "Interview_Score"
]

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Aurora PG College Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Aurora PG College")
st.subheader("Students Placement Prediction System")
st.markdown(
    "Predict student placement status and expected salary package "
    "using Machine Learning."
)

# -------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------
st.sidebar.title("🎯 Control Panel")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🤖 Model Training",
        "📈 Model Performance"
    ]
)

st.sidebar.markdown("---")

# -------------------------------------------------------
# LOAD TRAINING DATASET
# -------------------------------------------------------
@st.cache_data
def load_training_data():
    return pd.read_excel("Company_Placement_Dataset.xlsx")

def normalize(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def find_column(df, target):
    """Find a column matching `target` regardless of case/spacing/underscores."""
    target_norm = normalize(target)
    for col in df.columns:
        if normalize(col) == target_norm:
            return col
    return None

def auto_rename_columns(df):
    """Rename df columns in-place to match expected FEATURES/target names
    if a close match is found (handles case/space/underscore differences)."""
    rename_map = {}
    expected = FEATURES + ["Placement_Status", "Salary_LPA", "Student_ID"]
    for exp in expected:
        found = find_column(df, exp)
        if found and found != exp:
            rename_map[found] = exp
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

try:
    df = load_training_data()
    if df is not None:
        df = auto_rename_columns(df)
except Exception as e:
    df = None
    st.error(f"❌ Could not load training dataset: {e}")

# -------------------------------------------------------
# LOAD UPLOADED STUDENT DATA
# -------------------------------------------------------
def load_uploaded(file):
    if file is None:
        return None
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        else:
            st.error("Please upload a CSV or Excel (.xlsx/.xls) file.")
            return None
    except Exception as e:
        st.error(f"❌ Could not read uploaded file: {e}")
        return None

student_dataset = st.sidebar.file_uploader(
    "Upload Student Dataset",
    type=["csv", "xlsx", "xls"]
)

# Persist uploaded data in session_state so it survives page switches
if student_dataset is not None:
    parsed = load_uploaded(student_dataset)
    if parsed is not None:
        parsed = auto_rename_columns(parsed)
        st.session_state["uploaded_df1"] = parsed
        st.session_state["uploaded_df1_name"] = student_dataset.name

df1 = st.session_state.get("uploaded_df1", None)

st.sidebar.markdown("---")
st.sidebar.info(
"""
Features Used:

✔ CGPA
✔ Coding Score
✔ Aptitude Score
✔ Communication Skills
✔ Projects
✔ Certifications
✔ Internship
✔ Attendance
✔ Interview Score
"""
)

# -------------------------------------------------------
# SIDEBAR STATUS PANEL (diagnostics)
# -------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🩺 Status")
st.sidebar.write("Training data loaded:", df is not None)
st.sidebar.write("Student file uploaded:", df1 is not None)
st.sidebar.write("Model trained:", "classifier" in st.session_state)
st.sidebar.write("Predictions ready:", "prediction_df" in st.session_state)

with st.sidebar.expander("🔍 Debug: Training dataset columns"):
    if df is not None:
        st.write(list(df.columns))
    else:
        st.write("No training dataset loaded.")

# =========================================================
# DASHBOARD
# =========================================================
if page == "🏠 Dashboard":

    st.header("📊 Placement Prediction Dashboard")
    st.subheader("📘 Prediction Summary")

    if "prediction_df" in st.session_state:
        result = st.session_state["prediction_df"]
        total_students = len(result)
        placed = (result["Predicted_Placement"] == "Placed").sum()
        not_placed = (result["Predicted_Placement"] == "Not Placed").sum()
        placement_percentage = (placed / total_students) * 100 if total_students else 0
        avg_salary = result["Predicted_Salary"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", total_students)
        c2.metric("Placed", placed)
        c3.metric("Not Placed", not_placed)
        c4.metric("Placement %", f"{placement_percentage:.2f}%")
        st.metric("Average Salary", f"{avg_salary:.2f} LPA")

        st.markdown("---")
        st.subheader("Prediction Results")
        st.dataframe(result, use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Prediction Results",
            csv,
            "placement_predictions.csv",
            "text/csv"
        )
    else:
        st.warning("No predictions yet. Go to 🤖 Model Training to train the model "
                    "and generate predictions.")

    st.markdown("---")

    if df is not None:
        st.subheader("Company Training Dataset")
        tab1, tab2, tab3 = st.tabs(["Training Dataset", "Statistics", "Preview"])
        with tab1:
            st.dataframe(df, use_container_width=True)
        with tab2:
            st.dataframe(df.describe(include="all"), use_container_width=True)
        with tab3:
            st.dataframe(df.head())
    else:
        st.error("Training dataset not found.")

    st.markdown("---")
    st.subheader("📄 Uploaded Student Dataset")

    if df1 is not None:
        st.success(f"Student dataset '{st.session_state.get('uploaded_df1_name','')}' uploaded successfully")
        st.write(f"Total Uploaded Students : {len(df1)}")
        st.dataframe(df1, use_container_width=True)
    else:
        st.warning("Please upload a student dataset from the sidebar.")

# =========================================================
# DATA ANALYSIS
# =========================================================
elif page == "📊 Data Analysis":
    st.header("📊 Student Data Analysis")

    result1 = None
    if "prediction_df" in st.session_state:
        result1 = st.session_state["prediction_df"]
    elif os.path.exists("student_data.xlsx"):
        result1 = pd.read_excel("student_data.xlsx")

    if result1 is not None:

        st.subheader("🎓 Student Prediction Table")
        st.dataframe(result1, use_container_width=True)

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        total = len(result1)
        placed = (result1["Predicted_Placement"] == "Placed").sum()
        not_placed = (result1["Predicted_Placement"] == "Not Placed").sum()
        avg_salary = result1["Predicted_Salary"].mean()

        c1.metric("Total Students", total)
        c2.metric("Placed", placed)
        c3.metric("Not Placed", not_placed)
        c4.metric("Average Salary", f"{avg_salary:.2f} LPA")

        st.markdown("---")
        st.subheader("📋 Student Placement Report")

        report_cols = [c for c in FEATURES + [
            "Student_ID", "Predicted_Placement", "Predicted_Salary"
        ] if c in result1.columns]
        report = result1[report_cols]
        st.dataframe(report, use_container_width=True)

        csv = report.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Student Prediction Report",
            csv,
            "student_prediction_report.csv",
            "text/csv"
        )

        st.markdown("---")
        st.header("📈 Student Prediction Analytics")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                result1,
                names="Predicted_Placement",
                title="Predicted Placement Percentage",
                hole=0.45,
                color="Predicted_Placement",
                color_discrete_map={"Placed": "green", "Not Placed": "red"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                result1, x="Predicted_Salary", nbins=20,
                title="Predicted Salary Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            fig = px.box(
                result1, x="Predicted_Placement", y="CGPA",
                color="Predicted_Placement", title="CGPA vs Predicted Placement"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.box(
                result1, x="Predicted_Placement", y="Coding_Score",
                color="Predicted_Placement", title="Coding Score vs Predicted Placement"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col5, col6 = st.columns(2)
        with col5:
            fig = px.box(
                result1, x="Predicted_Placement", y="Interview_Score",
                color="Predicted_Placement", title="Interview Score Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            fig = px.histogram(
                result1, x="Attendance", color="Predicted_Placement",
                title="Attendance Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Predicted Salary by Student")
        fig = px.bar(
            result1.sort_values("Predicted_Salary", ascending=False),
            x="Student_ID", y="Predicted_Salary", color="Predicted_Placement",
            title="Expected Salary for Each Student"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠ Please train the model first to view student predictions.")
        st.info("Go to **🤖 Model Training**, make sure a student dataset is "
                "uploaded in the sidebar, then click **🚀 Train Models**.")

# =========================================================
# MODEL TRAINING
# =========================================================
elif page == "🤖 Model Training":
    st.header("🤖 Train Machine Learning Model")

    if df is None:
        st.error("Training dataset not found.")
    else:
        st.success("Training Dataset Loaded Successfully")
        st.write("Shape:", df.shape)

        missing_train_cols = [c for c in FEATURES + ["Placement_Status", "Salary_LPA"]
                               if c not in df.columns]
        if missing_train_cols:
            st.error(f"❌ Training dataset is missing required column(s): {missing_train_cols}")
        else:
            if st.button("🚀 Train Models"):
                data = df.copy()

                # Encode Internship
                data["Internship"] = (
                    data["Internship"].astype(str).str.strip().str.capitalize()
                    .map({"Yes": 1, "No": 0})
                )

                le = LabelEncoder()
                data["Placement_Status"] = le.fit_transform(data["Placement_Status"])

                X = data[FEATURES]
                y = data["Placement_Status"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.20, random_state=42
                )

                classifier = DecisionTreeClassifier(random_state=42)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                y_salary = data["Salary_LPA"]
                X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                    X, y_salary, test_size=0.20, random_state=42
                )

                regressor = DecisionTreeRegressor(random_state=42)
                regressor.fit(X_train_s, y_train_s)
                salary_pred = regressor.predict(X_test_s)

                mae = mean_absolute_error(y_test_s, salary_pred)
                r2 = r2_score(y_test_s, salary_pred)

                joblib.dump(classifier, "placement_model.pkl")
                joblib.dump(regressor, "salary_model.pkl")
                joblib.dump(le, "label_encoder.pkl")

                st.session_state["classifier"] = classifier
                st.session_state["regressor"] = regressor
                st.session_state["label_encoder"] = le
                st.session_state["features"] = FEATURES

                st.success("✅ Models Trained Successfully")

                c1, c2, c3 = st.columns(3)
                c1.metric("Classification Accuracy", f"{accuracy*100:.2f}%")
                c2.metric("Regression R² Score", f"{r2:.3f}")
                c3.metric("Mean Absolute Error", f"{mae:.2f}")

                st.markdown("---")

                importance = pd.DataFrame({
                    "Feature": FEATURES,
                    "Importance": classifier.feature_importances_
                }).sort_values("Importance", ascending=False)

                fig = px.bar(
                    importance, x="Importance", y="Feature", orientation="h",
                    color="Importance", title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(importance, use_container_width=True)

                st.markdown("---")

                # ---- Predict uploaded student dataset ----
                if df1 is None:
                    st.info("No student dataset uploaded yet. Upload one in the "
                            "sidebar — the model is trained and saved, so you "
                            "don't need to retrain; just re-open this page after "
                            "uploading.")
                else:
                    student = df1.copy()
                    missing_cols = [c for c in FEATURES if c not in student.columns]

                    if missing_cols:
                        st.error(
                            f"❌ Uploaded student dataset is missing required "
                            f"column(s): {missing_cols}. Expected: {FEATURES}"
                        )
                    else:
                        student["Internship"] = (
                            student["Internship"].astype(str).str.strip()
                            .str.capitalize().map({"Yes": 1, "No": 0})
                        )

                        if student["Internship"].isna().any():
                            st.warning("⚠ Some rows had an invalid 'Internship' "
                                       "value (not Yes/No) — those rows were dropped.")
                            student = student.dropna(subset=["Internship"])

                        if student.empty:
                            st.error("❌ No valid rows left to predict after cleaning.")
                        else:
                            try:
                                X_student = student[FEATURES]
                                placement_prediction = classifier.predict(X_student)
                                salary_prediction = regressor.predict(X_student)

                                student["Predicted_Placement"] = le.inverse_transform(
                                    placement_prediction
                                )
                                student["Predicted_Salary"] = salary_prediction.round(2)

                                st.session_state["prediction_df"] = student
                                student.to_excel("student_data.xlsx", index=False)

                                st.success("✅ Prediction completed and saved successfully.")
                                st.write("Prediction rows:", len(student))

                                st.markdown("---")
                                st.subheader("🎯 Prediction Results")
                                st.dataframe(student, use_container_width=True)

                                csv = student.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "📥 Download Prediction Results",
                                    csv,
                                    "placement_predictions.csv",
                                    "text/csv"
                                )
                            except Exception as e:
                                st.error(f"❌ Prediction failed: {e}")

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Dashboard")

    if df is None:
        st.warning("Please Upload Student Dataset")
    else:
        missing_perf_cols = [c for c in FEATURES + ["Placement_Status", "Salary_LPA"]
                              if c not in df.columns]
        if missing_perf_cols:
            st.error(
                f"❌ Training dataset is missing required column(s): {missing_perf_cols}"
            )
            st.info(f"Columns found in dataset: {list(df.columns)}")
            st.stop()

        data = df.copy()
        data["Internship"] = (
            data["Internship"].astype(str).str.strip().str.capitalize()
            .map({"Yes": 1, "No": 0})
        )

        le = LabelEncoder()
        data["Placement_Status"] = le.fit_transform(data["Placement_Status"])

        X = data[FEATURES]
        y = data["Placement_Status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction)
        recall = recall_score(y_test, prediction)
        f1 = f1_score(y_test, prediction)

        y_salary = data["Salary_LPA"]
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y_salary, test_size=0.20, random_state=42
        )

        regressor = RandomForestRegressor(n_estimators=200, random_state=42)
        regressor.fit(X_train2, y_train2)
        salary_pred = regressor.predict(X_test2)

        mae = mean_absolute_error(y_test2, salary_pred)
        rmse = np.sqrt(mean_squared_error(y_test2, salary_pred))
        r2 = r2_score(y_test2, salary_pred)

        st.subheader("Classification Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy*100:.2f}%")
        c2.metric("Precision", f"{precision:.2f}")
        c3.metric("Recall", f"{recall:.2f}")
        c4.metric("F1 Score", f"{f1:.2f}")

        st.markdown("---")
        st.subheader("Regression Metrics")
        d1, d2, d3 = st.columns(3)
        d1.metric("MAE", f"{mae:.2f}")
        d2.metric("RMSE", f"{rmse:.2f}")
        d3.metric("R² Score", f"{r2:.2f}")

        st.markdown("---")
        cm = confusion_matrix(y_test, prediction)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Classification Report")
        report = classification_report(y_test, prediction, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        st.markdown("---")
        result = pd.DataFrame({
            "Actual Salary": y_test2,
            "Predicted Salary": salary_pred
        })
        fig = px.scatter(result, x="Actual Salary", y="Predicted Salary",
                          title="Actual vs Predicted Salary")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": classifier.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                     color="Importance", title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🎓 Aurora PG College | Students Placement Prediction System | "
           "Developed using Streamlit & Machine Learning")
