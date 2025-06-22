import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import shap
import pycaret

token = st.secrets["DAGSHUB_TOKEN"]
dagshub.auth.add_app_token(token=token)

dagshub.init(repo_owner='oliviaosterlund', repo_name='finalprojectapp', mlflow=True)

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics

st.set_page_config(
    page_title="CO2 Emissions Predictor",
    layout="centered",
    page_icon="🚙",
)

df = pd.read_csv("CO2_Emission_Canada.csv")
st.sidebar.title("CO2 Emissions Predictor")
page = st.sidebar.selectbox("Select Page",["Introduction","Data Visualization", "Automated Report","Predictions"])

if page == "Introduction":
    st.title("CO2 Emissions Predictor")
    st.subheader("Analyzing the features of vehicles on CO2 emissions")
    st.markdown("""
    #### What this app does:
    - *Analyzes* key features
    - *Visualizes* trends and provides actionable insights
    - *Predicts* CO2 emissions using a variety of regression models
    """)
    st.image("co2car.jpg", width=500)

    st.markdown("#### The Dataset")
    st.markdown("""
    This dataset was collected to study the relationship between *student lifestyle and academic habits on performance outcomes (exam scores)*. It includes variables such as:
    - 📚 Class attendance and time studying
    - 🧠 Mental health
    - 💤 Sleep habits
    - 🍎 Diet
    - 🏃‍♂️ Exercise/physical activity 
    - 👩‍💻 Social media and netflix usage 
                
    The goal is to leverage these factors to *predict the academic success* in students and better understand
    which variables are most impactful.
    """)
    
    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("#####  Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Data Visualization":
    st.subheader("Data Viz")

    df_numeric = df.select_dtypes(include=np.number)

    tab1, tab2, tab3 = st.tabs(["Scatter Plot","Box Plot", "Correlation Heatmap"])
    with tab1:
        st.subheader("Scatter Plot")
        fig_bar, ax_bar = plt.subplots(figsize=(12,6))
        x_col = st.selectbox("Select x-axis variable", df_numeric.columns.drop(["exam_score","age", "exercise_frequency", "mental_health_rating"]))
        sns.scatterplot(df, x = x_col, y = "exam_score")
        st.pyplot(fig_bar)
    with tab2:
        st.subheader("Box Plot")
        fig_bar, ax_bar = plt.subplots(figsize=(12,6))
        x_col_2 = st.selectbox("Select x-axis variable", options=["exercise_frequency", "diet_quality", "mental_health_rating"])
        df_plot = df.copy()
        if x_col_2 == "diet_quality":
            df_plot['diet_quality'] = pd.Categorical(df_plot['diet_quality'], categories=['Poor', 'Fair', 'Good'], ordered=True)
            order = ['Poor', 'Fair', 'Good']
        else:
            order = None
        sns.boxplot(data=df_plot, x = x_col_2, y = "exam_score", order = order)
        st.pyplot(fig_bar)
    with tab3:
        st.subheader("Correlation Matrix")

        fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='coolwarm')
        
        st.pyplot(fig_corr)

elif page == "Automated Report":
    st.subheader("Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df,title="Student Habits vs Student Performance",explorative=True,minimal=True)
            st_profile_report(profile)
        export = profile.to_html()
        st.download_button(label="📥 Download full Report",data=export,file_name="student_habits_performance.html",mime='text/html')

elif page == "Predictions":
    st.subheader("Predictions")

    df2 = df.drop(["student_id","gender", "age", "parental_education_level", "internet_quality"], axis = 1)
    df2['diet_quality'] = df2['diet_quality'].map({'Poor': 0, 'Fair': 1, 'Good': 2})
    le = LabelEncoder()
    list_non_num =["part_time_job","extracurricular_participation"]
    for element in list_non_num:
        df2[element]= le.fit_transform(df2[element])
    
    list_var = list(df2.columns.drop("exam_score"))
    features_selection = st.sidebar.multiselect("Select Features (X)",list_var,default=list_var)
    if not features_selection:
        st.warning("Please select at least one feature")
        st.stop()
    
    model_name = st.sidebar.selectbox(
        "Choose Model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
    )

    params = {}
    if model_name == "Decision Tree":
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "XGBoost":
        params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)

    selected_metrics = st.sidebar.multiselect("Metrics to display", ["Mean Squared Error (MSE)","Mean Absolute Error (MAE)","R² Score"],default=["Mean Absolute Error (MAE)"])

    
    X = df2[features_selection]
    y = df2["exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)

    
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(**params, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror', **params, random_state=42)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = metrics.mean_squared_error(y_test, predictions)
        mae = metrics.mean_absolute_error(y_test, predictions)
        r2 = metrics.r2_score(y_test, predictions)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

    if "Mean Squared Error (MSE)" in selected_metrics:
        mse = metrics.mean_squared_error(y_test, predictions)
        st.write(f"- **MSE** {mse:,.2f}")
    if "Mean Absolute Error (MAE)" in selected_metrics:
        mae = metrics.mean_absolute_error(y_test, predictions)
        st.write(f"- **MAE** {mae:,.2f}")
    if "R² Score" in selected_metrics:
        r2 = metrics.r2_score(y_test, predictions)
        st.write(f"- **R2** {r2:,.3f}")
    
    fig, ax = plt.subplots()
    ax.scatter(y_test,predictions,alpha=0.5)
    ax.plot([y_test.min(),y_test.max()],
           [y_test.min(),y_test.max() ],"--r",linewidth=2)
    ax.set_xlabel("Actual Exam Scores")
    ax.set_ylabel("Predicted Exam Scores")
    ax.set_title("Actual vs Predicted Exam Scores")
    st.pyplot(fig)
