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

df = pd.read_csv("CO2_Emissions_Canada.csv")

df_numeric = df.select_dtypes(include=np.number)

df2 = df.dropna()
le = LabelEncoder()
list_non_num =["Make", "Model", "Vehicle Class", "Transmission", "Fuel Type"]
for element in list_non_num:
    df2[element]= le.fit_transform(df2[element])
    

st.sidebar.title("CO2 Emissions Predictor")
page = st.sidebar.selectbox("Select Page",["Introduction","Data Visualization", "Automated Report","Predictions", "Explainability"])

if page == "Introduction":
    st.title("CO2 Emissions Predictor")
    st.subheader("Analyzing the features of vehicles on CO2 emissions")
    st.markdown("""
    #### What this app does:
    - *Analyzes* key features
    - *Visualizes* trends and provides actionable insights into what features are the most impactful
    - *Predicts* CO2 emissions using a variety of regression models
    """)
    st.image("co2car.jpg", width=500)

    st.markdown("#### The Dataset")
    
    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display",5,20,5)
    st.dataframe(df.head(rows))

    st.markdown("#### Abbreviations Used in Dataset:")

    st.markdown("**Model:**")
    st.markdown("""
    - `4WD/4X4` – Four-wheel drive  
    - `AWD` – All-wheel drive  
    - `FFV` – Flexible-fuel vehicle  
    - `SWB` – Short wheelbase  
    - `LWB` – Long wheelbase  
    - `EWB` – Extended wheelbase  
    """)

    st.markdown("**Transmission:**")
    st.markdown("""
    - `A` – Automatic  
    - `AM` – Automated manual  
    - `AS` – Automatic with select shift  
    - `AV` – Continuously variable  
    - `M` – Manual  
    - `3`–`10` – Number of gears  
    """)

    st.markdown("**Fuel Type:**")
    st.markdown("""
    - `X` – Regular gasoline  
    - `Z` – Premium gasoline  
    - `D` – Diesel  
    - `E` – Ethanol (E85)  
    - `N` – Natural gas  
    """)
    
    st.markdown("#####  Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())

elif page == "Data Visualization":
    st.subheader("Data Viz")

    tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Scatter Plot", "Bar Chart", "Correlation Heatmap"])
    with tab1:
        st.subheader("Histogram")
        fig1, ax1 = plt.subplots()
        sns.histplot(df, x='CO2 Emissions(g/km)', binwidth=14, ax=ax1)
        ax1.set_title("Distribution of CO2 Emissions")
        ax1.set_ylabel('Frequency')
        st.pyplot(fig1)
    with tab2:
        st.subheader("Scatter Plot")
        col_x = st.selectbox("Select X-axis variable", df_numeric.columns.drop("CO2 Emissions(g/km)"), index=0)
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x=col_x, y="CO2 Emissions(g/km)", ax=ax2)
        ax2.set_title(f'{col_x} vs. CO2 Emissions')
        st.pyplot(fig2)
    with tab3:
        st.subheader("Bar Chart")
        cat_col = st.selectbox("Select a X-axis variable",["Make","Vehicle Class","Transmission","Fuel Type"])
        avg_emissions = df.groupby(cat_col)["CO2 Emissions(g/km)"].mean().sort_values(ascending=False)
        fig4, ax4 = plt.subplots()
        sns.barplot(x=avg_emissions.values, y=avg_emissions.index, ax=ax4, palette="cool")
        ax4.set_xlabel("Average CO2 Emissions (g/km)")
        ax4.set_title(f"Average CO2 Emissions by {cat_col}")
        st.pyplot(fig4)
    with tab4:
        st.subheader("Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(18,14))
        sns.heatmap(df_numeric.corr(),annot=True,fmt=".2f",cmap='cool')
        st.pyplot(fig_corr)

elif page == "Automated Report":
    st.subheader("Automated Report")
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            profile = ProfileReport(df,title="CO2 Emissions",explorative=True,minimal=True)
            st_profile_report(profile)
        export = profile.to_html()
        st.download_button(label="📥 Download full Report",data=export,file_name="student_habits_performance.html",mime='text/html')

elif page == "Predictions":
    st.subheader("Predictions")

    list_var = list(df2.columns.drop("CO2 Emissions(g/km)"))
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
    y = df2["CO2 Emissions(g/km)"]

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
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)
elif page == "Explainability":
    st.subheader("Explainability")
    # Load dataset
    X_shap, y_shap = df_numeric.drop(columns=["CO2 Emissions(g/km)"]), df_numeric["CO2 Emissions(g/km)"]
    # Train default XGBoost model for explainability
    model_exp = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model_exp.fit(X_shap, y_shap)

    # Create SHAP explainer and values
    explainer = shap.Explainer(model_exp)
    shap_values = explainer(X_shap)

    # SHAP Waterfall Plot for first prediction
    st.markdown("### SHAP Waterfall Plot for First Prediction")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())


    # SHAP Scatter Plot for 'Latitude'
    st.markdown("### SHAP Scatter Plot for 'Engine Size(L)'")
    shap.plots.scatter(shap_values[:, "Engine Size(L)"], color=shap_values, show=False)
    st.pyplot(plt.gcf())
