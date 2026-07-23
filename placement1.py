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
# MINIMAL CSS
# -------------------------------------------------------

st.markdown("""
<style>

.block-container{
    padding-top:1.5rem;
    padding-bottom:1rem;
}

[data-testid="stSidebar"]{
    background-color:#f7f7f7;
}

h1,h2,h3{
    color:#003366;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.title("🎓 Aurora PG College")
st.subheader("Students Placement Prediction System")

st.markdown("""
Predict student placement status and expected salary package
using Machine Learning.
""")

# -------------------------------------------------------
# LOAD DATASET FUNCTION
# -------------------------------------------------------

@st.cache_data
def load_data(student_dataset):

    if student_dataset is None:
        return None

    filename = student_dataset.name.lower()

    try:

        if filename.endswith(".csv"):
            df = pd.read_csv(student_dataset)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(student_dataset)

        else:
            st.error("Please upload only CSV or Excel files.")
            return None

        return df

    except Exception as e:
        st.error(f"Error reading file : {e}")
        return None

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

student_dataset = st.sidebar.file_uploader(
    "Upload Student Dataset",
    type=["csv", "xlsx", "xls"]
)

single_student = st.sidebar.file_uploader(
    "Upload Single Student File",
    type=["csv", "xlsx", "xls"]
)

df = None

if student_dataset is not None:
    df = load_data(student_dataset)

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Features Used

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

if page == "🏠 Dashboard":

    st.header("📋 Dashboard")

    if df is not None:

        st.success("Dataset Loaded Successfully")

        # -------------------------------
        # Dataset Preview
        # -------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.markdown("---")

        # -------------------------------
        # Metrics
        # -------------------------------

        total_students = len(df)

        if "Placement_Status" in df.columns:
            placed = len(df[df["Placement_Status"] == "Placed"])
            not_placed = len(df[df["Placement_Status"] == "Not Placed"])
            placement_percentage = (placed / total_students) * 100
        else:
            placed = 0
            not_placed = 0
            placement_percentage = 0

        if "Salary_LPA" in df.columns:
            avg_salary = df["Salary_LPA"].mean()
        else:
            avg_salary = 0

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Students",
            total_students
        )

        c2.metric(
            "Placed",
            placed
        )

        c3.metric(
            "Placement %",
            f"{placement_percentage:.2f}%"
        )

        c4.metric(
            "Average Salary",
            f"{avg_salary:.2f} LPA"
        )

        st.markdown("---")

        # -------------------------------
        # Tabs
        # -------------------------------

        tab1, tab2, tab3 = st.tabs(
            [
                "Dataset",
                "Statistics",
                "Columns"
            ]
        )

        with tab1:

            st.dataframe(
                df,
                use_container_width=True
            )

        with tab2:

            st.dataframe(
                df.describe(include="all"),
                use_container_width=True
            )

        with tab3:

            column_df = pd.DataFrame(
                {
                    "Column Name": df.columns,
                    "Data Type": df.dtypes.astype(str)
                }
            )

            st.dataframe(
                column_df,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("Last 10 Students")

        st.dataframe(
            df.tail(10),
            use_container_width=True
        )

    else:

        st.warning("Please upload Student Dataset from the sidebar.")
# -------------------------------------------------------
# DATA ANALYSIS
# -------------------------------------------------------

elif page == "📊 Data Analysis":

    st.header("📊 Student Placement Data Analysis")

    if df is not None:

        st.success("Dataset Loaded Successfully")

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        # Placement Distribution
        with col1:

            if "Placement_Status" in df.columns:

                fig = px.pie(
                    df,
                    names="Placement_Status",
                    title="Placement Distribution",
                    hole=0.45
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Salary Distribution
        with col2:

            if "Salary_LPA" in df.columns:

                fig = px.histogram(
                    df,
                    x="Salary_LPA",
                    nbins=20,
                    title="Salary Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.markdown("---")

        col3, col4 = st.columns(2)

        # CGPA Analysis
        with col3:

            if "CGPA" in df.columns and "Placement_Status" in df.columns:

                fig = px.box(
                    df,
                    x="Placement_Status",
                    y="CGPA",
                    color="Placement_Status",
                    title="CGPA Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Coding Score
        with col4:

            if "Coding_Score" in df.columns and "Placement_Status" in df.columns:

                fig = px.box(
                    df,
                    x="Placement_Status",
                    y="Coding_Score",
                    color="Placement_Status",
                    title="Coding Score Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.markdown("---")

        col5, col6 = st.columns(2)

        # Interview Score
        with col5:

            if "Interview_Score" in df.columns and "Placement_Status" in df.columns:

                fig = px.box(
                    df,
                    x="Placement_Status",
                    y="Interview_Score",
                    color="Placement_Status",
                    title="Interview Score Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Communication Skills
        with col6:

            if "Communication_Skills" in df.columns and "Placement_Status" in df.columns:

                fig = px.box(
                    df,
                    x="Placement_Status",
                    y="Communication_Skills",
                    color="Placement_Status",
                    title="Communication Skills Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.markdown("---")

        col7, col8 = st.columns(2)

        # Attendance
        with col7:

            if "Attendance" in df.columns:

                fig = px.histogram(
                    df,
                    x="Attendance",
                    color="Placement_Status",
                    title="Attendance Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # Internship
        with col8:

            if "Internship" in df.columns:

                internship = (
                    df.groupby("Internship")
                    .size()
                    .reset_index(name="Students")
                )

                fig = px.bar(
                    internship,
                    x="Internship",
                    y="Students",
                    title="Internship Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.markdown("---")

        st.subheader("Correlation Heatmap")

        numeric_df = df.select_dtypes(include=np.number)

        if len(numeric_df.columns) > 1:

            fig = px.imshow(
                numeric_df.corr(),
                text_auto=".2f",
                color_continuous_scale="RdBu"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("Dataset Summary")

        summary = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        })

        st.dataframe(
            summary,
            use_container_width=True
        )

    else:

        st.warning("Please upload Student Dataset.")

# -------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------

elif page == "🤖 Model Training":

    st.header("🤖 Train Placement Prediction Models")

    if df is not None:

        st.success("Dataset Loaded Successfully")

        st.write("Dataset Shape :", df.shape)

        st.markdown("---")

        train_btn = st.button("🚀 Train Placement Model")

        if train_btn:

            data = df.copy()

            # ---------------------------------------
            # Encode Internship
            # ---------------------------------------

            if "Internship" in data.columns:

                data["Internship"] = (
                    data["Internship"]
                    .astype(str)
                    .str.strip()
                    .str.title()
                    .map({
                        "Yes":1,
                        "No":0
                    })
                )

                data["Internship"] = data["Internship"].fillna(0)

            # ---------------------------------------
            # Encode Placement Status
            # ---------------------------------------

            placement_encoder = LabelEncoder()

            data["Placement_Status"] = placement_encoder.fit_transform(
                data["Placement_Status"]
            )

            # ---------------------------------------
            # Features
            # ---------------------------------------

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

            X = data[features]

            y = data["Placement_Status"]

            # ---------------------------------------
            # Train Test Split
            # ---------------------------------------

            X_train, X_test, y_train, y_test = train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42

            )

            # ---------------------------------------
            # Random Forest Classifier
            # ---------------------------------------

            classifier = RandomForestClassifier(

                n_estimators=200,

                random_state=42

            )

            classifier.fit(

                X_train,

                y_train

            )

            prediction = classifier.predict(

                X_test

            )

            # ---------------------------------------
            # Metrics
            # ---------------------------------------

            accuracy = accuracy_score(

                y_test,

                prediction

            )

            precision = precision_score(

                y_test,

                prediction

            )

            recall = recall_score(

                y_test,

                prediction

            )

            f1 = f1_score(

                y_test,

                prediction

            )

            # ---------------------------------------
            # Save Model
            # ---------------------------------------

            joblib.dump(

                classifier,

                "placement_model.pkl"

            )

            joblib.dump(

                placement_encoder,

                "placement_encoder.pkl"

            )

            # ---------------------------------------
            # Results
            # ---------------------------------------

            st.success("✅ Placement Model Trained Successfully")

            st.markdown("---")

            c1, c2, c3, c4 = st.columns(4)

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

            st.subheader("Confusion Matrix")

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

    else:

        st.warning("Please Upload Student Dataset")

        st.markdown("---")

            # =====================================================
            # SALARY PREDICTION MODEL
            # =====================================================

            st.subheader("💰 Salary Prediction Model")

            X_salary = data[features]

            y_salary = data["Salary_LPA"]

            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(

                X_salary,

                y_salary,

                test_size=0.20,

                random_state=42

            )

            regressor = RandomForestRegressor(

                n_estimators=200,

                random_state=42

            )

            regressor.fit(

                X_train_s,

                y_train_s

            )

            salary_prediction = regressor.predict(

                X_test_s

            )

            # ---------------------------------------
            # Regression Metrics
            # ---------------------------------------

            mae = mean_absolute_error(

                y_test_s,

                salary_prediction

            )

            mse = mean_squared_error(

                y_test_s,

                salary_prediction

            )

            rmse = np.sqrt(mse)

            r2 = r2_score(

                y_test_s,

                salary_prediction

            )

            # ---------------------------------------
            # Save Salary Model
            # ---------------------------------------

            joblib.dump(

                regressor,

                "salary_model.pkl"

            )

            st.success("✅ Salary Prediction Model Trained Successfully")

            d1, d2, d3 = st.columns(3)

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

            # =====================================================
            # FEATURE IMPORTANCE
            # =====================================================

            st.subheader("📊 Feature Importance")

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

            st.markdown("---")

            # =====================================================
            # TRAINING SUMMARY
            # =====================================================

            st.subheader("📋 Training Summary")

            summary = pd.DataFrame({

                "Model": [

                    "Placement Prediction",

                    "Salary Prediction"

                ],

                "Algorithm": [

                    "Random Forest Classifier",

                    "Random Forest Regressor"

                ],

                "Status": [

                    "Trained",

                    "Trained"

                ]

            })

            st.dataframe(

                summary,

                use_container_width=True

            )

            st.success("🎉 All Models Trained Successfully")
      # -------------------------------------------------------
# SINGLE PREDICTION
# -------------------------------------------------------

elif page == "🎯 Single Prediction":

    st.header("🎯 Single Student Placement Prediction")

    if single_student is not None:

        try:

            # -----------------------------------
            # Read Student File
            # -----------------------------------

            if single_student.name.endswith(".csv"):

                single_df = pd.read_csv(single_student)

            elif single_student.name.endswith(".xlsx"):

                single_df = pd.read_excel(single_student)

            elif single_student.name.endswith(".xls"):

                single_df = pd.read_excel(single_student)

            else:

                st.error("Upload CSV or Excel File")

                st.stop()

            st.success("Student File Uploaded Successfully")

            st.subheader("Student Details")

            st.dataframe(

                single_df,

                use_container_width=True

            )

            st.markdown("---")

            predict_btn = st.button("🚀 Predict Student")

            if predict_btn:

                # -----------------------------------
                # Load Models
                # -----------------------------------

                classifier = joblib.load("placement_model.pkl")

                regressor = joblib.load("salary_model.pkl")

                # -----------------------------------
                # Encode Internship
                # -----------------------------------

                single_df["Internship"] = (

                    single_df["Internship"]

                    .astype(str)

                    .str.strip()

                    .str.title()

                    .map({

                        "Yes":1,

                        "No":0

                    })

                )

                single_df["Internship"] = single_df["Internship"].fillna(0)

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

                X = single_df[features]

                # -----------------------------------
                # Placement Prediction
                # -----------------------------------

                placement_prediction = classifier.predict(X)

                probability = classifier.predict_proba(X)

                confidence = probability.max(axis=1) * 100

                # -----------------------------------
                # Salary Prediction
                # -----------------------------------

                salary_prediction = regressor.predict(X)

                # -----------------------------------
                # Store Results
                # -----------------------------------

                single_df["Predicted_Placement"] = np.where(

                    placement_prediction == 1,

                    "Placed",

                    "Not Placed"

                )

                single_df["Expected_Salary_LPA"] = salary_prediction.round(2)

                single_df["Confidence"] = confidence.round(2)
                              st.markdown("---")

                st.subheader("Prediction Results")

                status = single_df.loc[0, "Predicted_Placement"]
                salary = single_df.loc[0, "Expected_Salary_LPA"]
                confidence_score = single_df.loc[0, "Confidence"]

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Placement Status",
                    status
                )

                c2.metric(
                    "Expected Salary",
                    f"{salary:.2f} LPA"
                )

                c3.metric(
                    "Confidence",
                    f"{confidence_score:.2f}%"
                )

                st.markdown("---")

                # -----------------------------------
                # Prediction Result Table
                # -----------------------------------

                st.subheader("Prediction Report")

                st.dataframe(
                    single_df,
                    use_container_width=True
                )

                st.markdown("---")

                # -----------------------------------
                # Confidence Gauge
                # -----------------------------------

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=confidence_score,
                        title={"text": "Prediction Confidence"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 40], "color": "red"},
                                {"range": [40, 70], "color": "orange"},
                                {"range": [70, 100], "color": "green"},
                            ],
                        },
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.markdown("---")

                # -----------------------------------
                # Download Prediction Report
                # -----------------------------------

                csv = single_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇ Download Prediction Report",
                    data=csv,
                    file_name="Single_Student_Prediction.csv",
                    mime="text/csv"
                )

                st.success("Prediction Completed Successfully")

        except Exception as e:

            st.error(f"Error : {e}")

    else:

        st.info("Please upload a Single Student CSV/Excel file.")

# -------------------------------------------------------
# BATCH PREDICTION
# -------------------------------------------------------

elif page == "📁 Batch Prediction":

    st.header("📁 Batch Student Placement Prediction")

    if df is not None:

        st.success("Dataset Loaded Successfully")

        if st.button("🚀 Predict All Students"):

            try:

                # ---------------------------------------
                # Load Models
                # ---------------------------------------

                classifier = joblib.load("placement_model.pkl")
                regressor = joblib.load("salary_model.pkl")

                batch_df = df.copy()

                # ---------------------------------------
                # Encode Internship
                # ---------------------------------------

                batch_df["Internship"] = (
                    batch_df["Internship"]
                    .astype(str)
                    .str.strip()
                    .str.title()
                    .map({
                        "Yes": 1,
                        "No": 0
                    })
                )

                batch_df["Internship"] = batch_df["Internship"].fillna(0)

                # ---------------------------------------
                # Features
                # ---------------------------------------

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

                # ---------------------------------------
                # Placement Prediction
                # ---------------------------------------

                placement_prediction = classifier.predict(X)

                probability = classifier.predict_proba(X)

                confidence = probability.max(axis=1) * 100

                # ---------------------------------------
                # Salary Prediction
                # ---------------------------------------

                salary_prediction = regressor.predict(X)

                # ---------------------------------------
                # Save Results
                # ---------------------------------------

                batch_df["Predicted_Placement"] = np.where(

                    placement_prediction == 1,

                    "Placed",

                    "Not Placed"

                )

                batch_df["Expected_Salary_LPA"] = salary_prediction.round(2)

                batch_df["Confidence"] = confidence.round(2)

                st.success("Batch Prediction Completed Successfully")

                st.markdown("---")

                st.subheader("Prediction Results")

                st.dataframe(

                    batch_df,

                    use_container_width=True

                )

            except Exception as e:

                st.error(e)

    else:

        st.warning("Please Upload Student Dataset")
                      st.markdown("---")

                # ==========================================
                # DASHBOARD METRICS
                # ==========================================

                total_students = len(batch_df)

                placed = len(
                    batch_df[
                        batch_df["Predicted_Placement"] == "Placed"
                    ]
                )

                not_placed = total_students - placed

                placement_percentage = (
                    placed / total_students
                ) * 100

                average_salary = (
                    batch_df["Expected_Salary_LPA"].mean()
                )

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Students",
                    total_students
                )

                c2.metric(
                    "Placed",
                    placed
                )

                c3.metric(
                    "Placement %",
                    f"{placement_percentage:.2f}%"
                )

                c4.metric(
                    "Average Salary",
                    f"{average_salary:.2f} LPA"
                )

                st.markdown("---")

                # ==========================================
                # PIE CHART
                # ==========================================

                col1, col2 = st.columns(2)

                with col1:

                    fig = px.pie(

                        batch_df,

                        names="Predicted_Placement",

                        hole=0.45,

                        title="Placement Distribution"

                    )

                    st.plotly_chart(

                        fig,

                        use_container_width=True

                    )

                # ==========================================
                # SALARY DISTRIBUTION
                # ==========================================

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

                # ==========================================
                # CONFIDENCE DISTRIBUTION
                # ==========================================

                col3, col4 = st.columns(2)

                with col3:

                    fig = px.histogram(

                        batch_df,

                        x="Confidence",

                        nbins=20,

                        title="Prediction Confidence"

                    )

                    st.plotly_chart(

                        fig,

                        use_container_width=True

                    )

                # ==========================================
                # PLACEMENT BAR CHART
                # ==========================================

                with col4:

                    summary = (

                        batch_df.groupby(
                            "Predicted_Placement"
                        )

                        .size()

                        .reset_index(name="Students")

                    )

                    fig = px.bar(

                        summary,

                        x="Predicted_Placement",

                        y="Students",

                        title="Placement Summary"

                    )

                    st.plotly_chart(

                        fig,

                        use_container_width=True

                    )

                st.markdown("---")

                # ==========================================
                # DOWNLOAD CSV
                # ==========================================

                csv = batch_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(

                    "⬇ Download Prediction Report",

                    csv,

                    "Batch_Prediction_Report.csv",

                    "text/csv"

                )

                st.success(
                    "Batch Prediction Completed Successfully"
                )
      # -------------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------------

elif page == "📈 Model Performance":

    st.header("📈 Model Performance Dashboard")

    if df is not None:

        try:

            # ---------------------------------------
            # Prepare Data
            # ---------------------------------------

            data = df.copy()

            data["Internship"] = (
                data["Internship"]
                .astype(str)
                .str.strip()
                .str.title()
                .map({"Yes": 1, "No": 0})
            )

            data["Internship"] = data["Internship"].fillna(0)

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

            # ---------------------------------------
            # Classification Model
            # ---------------------------------------

            encoder = LabelEncoder()

            y = encoder.fit_transform(data["Placement_Status"])

            X = data[features]

            X_train, X_test, y_train, y_test = train_test_split(

                X,

                y,

                test_size=0.20,

                random_state=42

            )

            classifier = RandomForestClassifier(

                n_estimators=200,

                random_state=42

            )

            classifier.fit(

                X_train,

                y_train

            )

            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            precision = precision_score(

                y_test,

                y_pred,

                zero_division=0

            )

            recall = recall_score(

                y_test,

                y_pred,

                zero_division=0

            )

            f1 = f1_score(

                y_test,

                y_pred,

                zero_division=0

            )

            # ---------------------------------------
            # Regression Model
            # ---------------------------------------

            y_salary = data["Salary_LPA"]

            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(

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

            mse = mean_squared_error(

                y_test_s,

                salary_pred

            )

            rmse = np.sqrt(mse)

            r2 = r2_score(

                y_test_s,

                salary_pred

            )

            # ---------------------------------------
            # Metrics
            # ---------------------------------------

            st.subheader("Classification Metrics")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(

                "Accuracy",

                f"{accuracy:.2%}"

            )

            c2.metric(

                "Precision",

                f"{precision:.2%}"

            )

            c3.metric(

                "Recall",

                f"{recall:.2%}"

            )

            c4.metric(

                "F1 Score",

                f"{f1:.2%}"

            )

            st.markdown("---")

            st.subheader("Regression Metrics")

            r1, r2c, r3 = st.columns(3)

            r1.metric(

                "MAE",

                f"{mae:.2f}"

            )

            r2c.metric(

                "RMSE",

                f"{rmse:.2f}"

            )

            r3.metric(

                "R² Score",

                f"{r2:.2f}"

            )

            st.markdown("---")

            # ---------------------------------------
            # Confusion Matrix
            # ---------------------------------------

            cm = confusion_matrix(

                y_test,

                y_pred

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

            # ---------------------------------------
            # Classification Report
            # ---------------------------------------

            st.subheader("Classification Report")

            report = classification_report(

                y_test,

                y_pred,

                output_dict=True

            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(

                report_df,

                use_container_width=True

            )

            st.markdown("---")

            # ---------------------------------------
            # Feature Importance
            # ---------------------------------------

            st.subheader("Feature Importance")

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

            st.markdown("---")

            # ---------------------------------------
            # Actual vs Predicted Salary
            # ---------------------------------------

            st.subheader("Actual vs Predicted Salary")

            comparison = pd.DataFrame({

                "Actual Salary": y_test_s.values,

                "Predicted Salary": salary_pred

            })

            fig = px.scatter(

                comparison,

                x="Actual Salary",

                y="Predicted Salary",

                title="Actual vs Predicted Salary"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

            st.dataframe(

                comparison,

                use_container_width=True

            )

            st.success("🎉 Model Performance Analysis Completed Successfully")

        except Exception as e:

            st.error(f"Error: {e}")

    else:

        st.info("Please upload the student dataset first.")

  
