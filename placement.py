# ==========================================================
# Aurora PG College Students Placement Prediction System
# Part - 1A
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Aurora PG College Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.title("🎓 Aurora PG College")
st.subheader("Students Placement Prediction System")

st.markdown(
"""
Predict student placement status and expected salary package
using Machine Learning.
"""
)

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("🎯 Control Panel")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🤖 Model Training",
        "🎯 Single Prediction",
        "📁 Batch Prediction",
        "📈 Model Performance"
    ]
)

st.sidebar.markdown("---")
# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------

@st.cache_data
def load_data(student_dataset):
    # student_dataset = st.sidebar.file_uploader(...)

    if student_dataset.name.endswith(".csv"):
        df1 = pd.read_csv(student_dataset)

    elif student_dataset.name.endswith(".xlsx"):
        df1 = pd.read_excel(student_dataset)

    elif student_dataset.name.endswith(".xls"):
        df1 = pd.read_excel(student_dataset)

    else:
        st.error("Please upload a CSV or Excel (.xlsx/.xls) file.")
        return None

    return df1

# Training Dataset (stored in GitHub)
df = pd.read_excel("Company_Placement_Dataset.xlsx")
# -------------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------------
# Training Dataset (stored in GitHub)
#df1 = pd.read_csv("Company_Placement_Dataset.csv")

student_dataset = st.sidebar.file_uploader(
    "Upload Student Dataset",
    type=["csv", "xlsx", "xls", "json", "txt", "tsv", "pdf", "pptx"]
)

single_student = st.sidebar.file_uploader(
    "Upload Single Student File",
    type=["csv", "xlsx", "xls", "json", "txt", "tsv", "pdf", "pptx"]
)

df1 = None

if student_dataset is not None:
    df1 = load_data(student_dataset)

#st.write(student_dataset)

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
# DASHBOARD
# -------------------------------------------------------
# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------
if page == "🏠 Dashboard":

    st.header("📊 Placement Prediction Dashboard")

    # ==========================
    # Training Dataset Dashboard
    # ==========================
    st.subheader("📘 Student Dataset")

    if "prediction_df" in st.session_state:
        
        result = st.session_state["prediction_df"]
        total_students = len(result)
        placed = (result["Predicted_Placement"] == "Placed").sum()
        not_placed = (result["Predicted_Placement"] == "Not Placed").sum()
        placement_percentage = (placed / total_students) * 100
        avg_salary = result["Predicted_Salary"].mean()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Students", total_students)
        c2.metric("Placed", placed)
        c3.metric("Not Placed", not_placed)
        c4.metric("Placement %", f"{placement_percentage:.2f}%")

        st.metric("Average Salary", f"{avg_salary:.2f} LPA")

        st.markdown("---")

        st.subheader("Prediction Results")

        st.dataframe(df1, use_container_width=True)

        csv = df1.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction Results",
            csv,
            "placement_predictions.csv",
            "text/csv"
        )

    else:
        st.warning("Please upload a student dataset.")
    if df is not None:
        
        st.subheader("Company Dataset")

        tab1, tab2, tab3 = st.tabs(
            ["Training Dataset", "Statistics", "Preview"]
        )

        with tab1:
            st.dataframe(df, use_container_width=True)

        with tab2:
            st.dataframe(df.describe(include="all"),
                         use_container_width=True)

        with tab3:
            st.dataframe(df.head())

    else:
        st.error("Training dataset not found.")

    st.markdown("---")

    # ==========================
    # Uploaded Student Dataset
    # ==========================
    st.subheader("📄 Uploaded Student Dataset")

    if df1 is not None:

        st.success("Student Dataset Uploaded Successfully")

        st.write(f"Total Uploaded Students : {len(df1)}")

        st.dataframe(df1, use_container_width=True)

    else:
        st.warning("Please upload a student dataset from the sidebar.")

# -------------------------------------------------------
# DATA ANALYSIS
# ------------------------------------------------------- 

elif page=="📊 Data Analysis":

    st.header("📊 Student Data Analysis")

    if "prediction_df" in st.session_state:

        result = st.session_state["prediction_df"]

        st.subheader("🎓 Student Prediction Table")

        st.dataframe(
            result,
            use_container_width=True
        )

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)

        total = len(result)

        placed = (result["Predicted_Placement"] == "Placed").sum()

        not_placed = (result["Predicted_Placement"] == "Not Placed").sum()

        avg_salary = result["Predicted_Salary"].mean()

        c1.metric("Total Students", total)
        c2.metric("Placed", placed)
        c3.metric("Not Placed", not_placed)
        c4.metric("Average Salary", f"{avg_salary:.2f} LPA")

        st.markdown("---")

        st.subheader("📋 Student Placement Report")

        report = result[[
            "Student_ID",
            "CGPA",
            "Coding_Score",
            "Aptitude_Score",
            "Communication_Skills",
            "Projects",
            "Certifications",
            "Internship",
            "Attendance",
            "Interview_Score",
            "Predicted_Placement",
            "Predicted_Salary"
        ]]

        st.dataframe(
            report,
            use_container_width=True
        )

        csv = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Student Prediction Report",
            csv,
            "student_prediction_report.csv",
            "text/csv"
        )

    else:

        st.warning("⚠ Please train the model first to view student predictions.")




# -------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------

elif page=="🤖 Model Training":

    st.header("🤖 Train Machine Learning Model")

    if df is not None:

        st.success("Dataset Loaded Successfully")

        st.write(df.shape)

    else:

        st.warning("Upload Dataset")

# -------------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------------

elif page=="🎯 Single Prediction":

    st.header("🎯 Single Student Prediction")

    if single_student is not None:

        single_df = pd.read_csv(single_student)

        st.dataframe(single_df)

    else:

        st.info("Upload Single Student CSV")

# -------------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------------

elif page=="📁 Batch Prediction":

    st.header("📁 Batch Prediction")

    st.info("Prediction will be here ")

# -------------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------------

elif page=="📈 Model Performance":

    st.header("📈 Model Performance")

    st.info("Accuracy, Confusion Matrix and Reports will be here")
# -------------------------------------------------------
# DATA ANALYSIS
# -------------------------------------------------------
elif page == "📊 Data Analysis":
    st.markdown("---")
    st.header("📈 Student Prediction Analytics")
    result = st.session_state["prediction_df"]
    col1, col2 = st.columns(2)

# Placement Pie Chart
with col1:

     fig = px.pie(
        result,
        names="Predicted_Placement",
        title="Predicted Placement Percentage",
        hole=0.45,
        color="Predicted_Placement",
        color_discrete_map={
            "Placed": "green",
            "Not Placed": "red"
        }
    )

  
# -------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------


elif page == "🤖 Model Training":

    st.header("🤖 Train Placement Prediction Models")

    if df is not None:

        if st.button("🚀 Train Models"):

            data = df.copy()

            # -----------------------------------
            # Encode Categorical Columns
            # -----------------------------------

            data["Internship"] = data["Internship"].map({
                "Yes": 1,
                "No": 0
            })

            le = LabelEncoder()

            data["Placement_Status"] = le.fit_transform(
                data["Placement_Status"]
            )

            # -----------------------------------
            # Features
            # -----------------------------------

            features = [
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

            # -----------------------------------
            # Classification Model
            # -----------------------------------

            X = data[features]
            y = data["Placement_Status"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )

            classifier = DecisionTreeClassifier(
                random_state=42
            )

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            # -----------------------------------
            # Regression Model
            # -----------------------------------

            X_salary = data[features]
            y_salary = data["Salary_LPA"]

            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
                X_salary,
                y_salary,
                test_size=0.20,
                random_state=42
            )

            regressor = DecisionTreeRegressor(
                random_state=42
            )

            regressor.fit(
                X_train_s,
                y_train_s
            )

            salary_pred = regressor.predict(
                X_test_s
            )

            mae = mean_absolute_error(
                y_test_s,
                salary_pred
            )

            r2 = r2_score(
                y_test_s,
                salary_pred
            )

            # -----------------------------------
            # Save Models
            # -----------------------------------

            joblib.dump(
                classifier,
                "placement_model.pkl"
            )

            joblib.dump(
                regressor,
                "salary_model.pkl"
            )

            joblib.dump(
                le,
                "label_encoder.pkl"
            )

            # Store in session state

            st.session_state["classifier"] = classifier
            st.session_state["regressor"] = regressor
            st.session_state["label_encoder"] = le
            st.session_state["features"] = features

            # -----------------------------------
            # Metrics
            # -----------------------------------

            st.success("✅ Models Trained Successfully")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Classification Accuracy",
                f"{accuracy*100:.2f}%"
            )

            c2.metric(
                "Regression R² Score",
                f"{r2:.3f}"
            )

            c3.metric(
                "Mean Absolute Error",
                f"{mae:.2f}"
            )

            st.markdown("---")

            # -----------------------------------
            # Feature Importance
            # -----------------------------------

            importance = pd.DataFrame({

                "Feature": features,

                "Importance": classifier.feature_importances_

            })

            importance = importance.sort_values(
                "Importance",
                ascending=False
            )

            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                title="Feature Importance"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance,
                use_container_width=True
            )

            # -----------------------------------
            # Predict Uploaded Dataset
            # -----------------------------------

            if df1 is not None:

                student = df1.copy()

                student["Internship"] = student["Internship"].map({
                    "Yes": 1,
                    "No": 0
                })

                X_student = student[features]

                placement_prediction = classifier.predict(X_student)

                salary_prediction = regressor.predict(X_student)

                student["Predicted_Placement"] = le.inverse_transform(
                    placement_prediction
                )

                student["Predicted_Salary"] = salary_prediction.round(2)

                st.session_state["prediction_df"] = student

                st.markdown("---")

                st.subheader("🎯 Prediction Results")

                st.dataframe(
                    student,
                    use_container_width=True
                )

                csv = student.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Download Prediction Results",
                    csv,
                    "placement_predictions.csv",
                    "text/csv"
                )

            else:

                st.info(
                    "Upload a student dataset to generate predictions."
                )

    else:

        st.error("Training dataset not found.")

# -------------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------------

elif page == "🎯 Single Prediction":

    st.header("🎯 Single Student Placement Prediction")

    if single_student is not None:

        try:

            single_df = pd.read_csv(single_student)

            st.subheader("Uploaded Student Details")

            st.dataframe(single_df, use_container_width=True)

            if st.button("Predict Student"):

                # Load Saved Models
                classifier = joblib.load("placement_model.pkl")
                regressor = joblib.load("salary_model.pkl")

                # Convert Internship
                single_df["Internship"] = single_df["Internship"].map({
                    "Yes":1,
                    "No":0
                })

                features = [
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

                X = single_df[features]

                # Placement Prediction
                placement = classifier.predict(X)

                probability = classifier.predict_proba(X)

                confidence = probability.max()*100

                # Salary Prediction
                salary = regressor.predict(X)

                status = "Placed" if placement[0]==1 else "Not Placed"

                st.markdown("---")

                c1,c2,c3 = st.columns(3)

                c1.metric(
                    "Placement Status",
                    status
                )

                c2.metric(
                    "Expected Salary",
                    f"{salary[0]:.2f} LPA"
                )

                c3.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.success("Prediction Completed Successfully")

                # Gauge Chart

                import plotly.graph_objects as go

                fig = go.Figure(go.Indicator(

                    mode="gauge+number",

                    value=confidence,

                    title={'text':"Prediction Confidence"},

                    gauge={

                        'axis':{'range':[0,100]},

                        'bar':{'color':'darkblue'},

                        'steps':[

                            {'range':[0,40],'color':'red'},

                            {'range':[40,70],'color':'orange'},

                            {'range':[70,100],'color':'green'}

                        ]

                    }

                ))

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as e:

            st.error(e)

    else:

        st.info("Upload Single Student CSV")
      # -------------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------------

elif page == "📁 Batch Prediction":

    st.header("📁 Batch Student Placement Prediction")

    if df is not None:

        if st.button("Predict All Students"):

            # Load Models
            classifier = joblib.load("placement_model.pkl")
            regressor = joblib.load("salary_model.pkl")

            batch_df = df.copy()

            # Encode Internship
            batch_df["Internship"] = batch_df["Internship"].map({
                "Yes":1,
                "No":0
            })

            features = [
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

            X = batch_df[features]

            # Placement Prediction
            placement = classifier.predict(X)

            probability = classifier.predict_proba(X)

            confidence = probability.max(axis=1) * 100

            # Salary Prediction
            salary = regressor.predict(X)

            batch_df["Predicted_Placement"] = np.where(
                placement==1,
                "Placed",
                "Not Placed"
            )

            batch_df["Expected_Salary_LPA"] = salary.round(2)

            batch_df["Confidence"] = confidence.round(2)

            st.success("Batch Prediction Completed")

            st.subheader("Prediction Results")

            st.dataframe(
                batch_df,
                use_container_width=True
            )

            # ============================
            # METRICS
            # ============================

            total = len(batch_df)

            placed = len(
                batch_df[
                    batch_df["Predicted_Placement"]=="Placed"
                ]
            )

            notplaced = total - placed

            placement_rate = placed/total*100

            avg_salary = batch_df["Expected_Salary_LPA"].mean()

            c1,c2,c3,c4 = st.columns(4)

            c1.metric("Students", total)

            c2.metric("Placed", placed)

            c3.metric(
                "Placement %",
                f"{placement_rate:.2f}%"
            )

            c4.metric(
                "Average Salary",
                f"{avg_salary:.2f} LPA"
            )

            st.markdown("---")

            col1,col2 = st.columns(2)

            # Placement Pie Chart
            with col1:

                fig = px.pie(
                    batch_df,
                    names="Predicted_Placement",
                    title="Predicted Placement Percentage",
                    hole=0.45
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # Salary Histogram
            with col2:

                fig = px.histogram(
                    batch_df,
                    x="Expected_Salary_LPA",
                    nbins=20,
                    title="Expected Salary Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.markdown("---")

            # Download CSV

            csv = batch_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Prediction CSV",
                csv,
                "Placement_Predictions.csv",
                "text/csv"
            )

    else:

        st.warning("Please Upload Student Dataset")
      # -------------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------------

elif page == "📈 Model Performance":

    st.header("📈 Model Performance Dashboard")

    if df is not None:

        data = df.copy()

        # Encode Internship
        data["Internship"] = data["Internship"].map({
            "Yes":1,
            "No":0
        })

        # Encode Placement Status
        le = LabelEncoder()

        data["Placement_Status"] = le.fit_transform(
            data["Placement_Status"]
        )

        features = [
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

        # -------------------------
        # Classification
        # -------------------------

        X = data[features]
        y = data["Placement_Status"]

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        classifier.fit(X_train,y_train)

        prediction = classifier.predict(X_test)

        accuracy = accuracy_score(y_test,prediction)
        precision = precision_score(y_test,prediction)
        recall = recall_score(y_test,prediction)
        f1 = f1_score(y_test,prediction)

        # -------------------------
        # Regression
        # -------------------------

        y_salary = data["Salary_LPA"]

        X_train2,X_test2,y_train2,y_test2 = train_test_split(
            X,
            y_salary,
            test_size=0.20,
            random_state=42
        )

        regressor = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        regressor.fit(
            X_train2,
            y_train2
        )

        salary_pred = regressor.predict(X_test2)

        mae = mean_absolute_error(
            y_test2,
            salary_pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test2,
                salary_pred
            )
        )

        r2 = r2_score(
            y_test2,
            salary_pred
        )

        # -------------------------
        # Metrics
        # -------------------------

        st.subheader("Classification Metrics")

        c1,c2,c3,c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{precision:.2f}"
        )

        c3.metric(
            "Recall",
            f"{recall:.2f}"
        )

        c4.metric(
            "F1 Score",
            f"{f1:.2f}"
        )

        st.markdown("---")

        st.subheader("Regression Metrics")

        d1,d2,d3 = st.columns(3)

        d1.metric(
            "MAE",
            f"{mae:.2f}"
        )

        d2.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        d3.metric(
            "R² Score",
            f"{r2:.2f}"
        )

        st.markdown("---")

        # -------------------------
        # Confusion Matrix
        # -------------------------

        cm = confusion_matrix(
            y_test,
            prediction
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        # -------------------------
        # Classification Report
        # -------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            prediction,
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df,
            use_container_width=True
        )

        st.markdown("---")

        # -------------------------
        # Actual vs Predicted Salary
        # -------------------------

        result = pd.DataFrame({

            "Actual Salary":y_test2,

            "Predicted Salary":salary_pred

        })

        fig = px.scatter(
            result,
            x="Actual Salary",
            y="Predicted Salary",
            title="Actual vs Predicted Salary"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        # -------------------------
        # Feature Importance
        # -------------------------

        importance = pd.DataFrame({

            "Feature":features,

            "Importance":classifier.feature_importances_

        })

        importance = importance.sort_values(
            "Importance",
            ascending=False
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning("Please Upload Student Dataset")

st.markdown("---")
st.caption("🎓 Aurora PG College | Students Placement Prediction System | Developed using Streamlit & Machine Learning")
