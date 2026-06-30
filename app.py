import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="AI Impact on Jobs 2030 Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Impact on Jobs 2030 Dashboard")

st.write(
    "This dashboard predicts AI Replacement Risk using the best Linear Regression model developed in Q2."
)

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data():
    return pd.read_csv("AI_Impact_on_Jobs_2030.csv")

df = load_data()

# ==========================================
# Dashboard Filters
# ==========================================

st.sidebar.header("Dashboard Filters")

selected_industry = st.sidebar.selectbox(
    "Filter by Industry",
    ["All"] + sorted(df["Industry"].unique().tolist())
)

risk_filter = st.sidebar.slider(
    "Maximum AI Replacement Risk",
    min_value=float(df["AI_Replacement_Risk"].min()),
    max_value=float(df["AI_Replacement_Risk"].max()),
    value=float(df["AI_Replacement_Risk"].max()),
    step=0.01
)

filtered_df = df.copy()

if selected_industry != "All":
    filtered_df = filtered_df[
        filtered_df["Industry"] == selected_industry
    ]

filtered_df = filtered_df[
    filtered_df["AI_Replacement_Risk"] <= risk_filter
]

# ==================================================
# LOAD TRAINED MODEL
# ==================================================
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("label_encoders.pkl")

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("Prediction Inputs")

# ----------------------------
# CATEGORICAL INPUTS
# ----------------------------

job_title = st.sidebar.selectbox(
    "Job Title",
    sorted(df["Job_Title"].unique())
)

industry = st.sidebar.selectbox(
    "Industry",
    sorted(df["Industry"].unique())
)

country = st.sidebar.selectbox(
    "Country",
    sorted(df["Country"].unique())
)

education = st.sidebar.selectbox(
    "Education Level",
    sorted(df["Education_Level"].unique())
)

remote = st.sidebar.selectbox(
    "Remote Work Possibility",
    sorted(df["Remote_Work_Possibility"].unique())
)

skills = st.sidebar.selectbox(
    "Required Skills",
    sorted(df["Required_Skills"].unique())
)

automation = st.sidebar.selectbox(
    "Automation Level",
    sorted(df["Automation_Level"].unique())
)

company = st.sidebar.selectbox(
    "Company Size",
    sorted(df["Company_Size"].unique())
)

ai_tool = st.sidebar.selectbox(
    "AI Tool Usage",
    sorted(df["AI_Tool_Usage"].unique())
)

upskill = st.sidebar.selectbox(
    "Upskilling Needed",
    sorted(df["Upskilling_Needed"].unique())
)

hiring = st.sidebar.selectbox(
    "Hiring Trend 2026",
    sorted(df["Hiring_Trend_2026"].unique())
)

# ----------------------------
# NUMERICAL INPUTS
# ----------------------------

years = st.sidebar.slider(
    "Years of Experience",
    min_value=int(df["Years_Experience"].min()),
    max_value=int(df["Years_Experience"].max()),
    value=int(df["Years_Experience"].median())
)

future = st.sidebar.number_input(
    "Future Demand Score",
    value=float(df["Future_Demand_Score"].mean())
)

salary = st.sidebar.number_input(
    "Average Salary (USD)",
    value=float(df["Average_Salary_USD"].mean())
)

growth = st.sidebar.number_input(
    "Job Growth 2030",
    value=float(df["Job_Growth_2030"].mean())
)

hours = st.sidebar.number_input(
    "Work Hours Per Week",
    value=float(df["Work_Hours_Per_Week"].mean())
)

performance = st.sidebar.number_input(
    "Performance Score",
    value=float(df["Performance_Score"].mean())
)

satisfaction = st.sidebar.number_input(
    "Job Satisfaction",
    value=float(df["Job_Satisfaction"].mean())
)

# ==================================================
# PREPARE USER INPUT
# ==================================================

# Encode categorical inputs
job_title_encoded = encoders["Job_Title"].transform([job_title])[0]
industry_encoded = encoders["Industry"].transform([industry])[0]
country_encoded = encoders["Country"].transform([country])[0]
education_encoded = encoders["Education_Level"].transform([education])[0]
remote_encoded = encoders["Remote_Work_Possibility"].transform([remote])[0]
skills_encoded = encoders["Required_Skills"].transform([skills])[0]
automation_encoded = encoders["Automation_Level"].transform([automation])[0]
company_encoded = encoders["Company_Size"].transform([company])[0]
ai_tool_encoded = encoders["AI_Tool_Usage"].transform([ai_tool])[0]
upskill_encoded = encoders["Upskilling_Needed"].transform([upskill])[0]
hiring_encoded = encoders["Hiring_Trend_2026"].transform([hiring])[0]

# Scale numerical variables
numerical_input = pd.DataFrame({
    "Years_Experience":[years],
    "Future_Demand_Score":[future],
    "Average_Salary_USD":[salary],
    "Job_Growth_2030":[growth],
    "Work_Hours_Per_Week":[hours],
    "Performance_Score":[performance],
    "Job_Satisfaction":[satisfaction]
})

scaled = scaler.transform(numerical_input)

years_scaled = scaled[0][0]
future_scaled = scaled[0][1]
salary_scaled = scaled[0][2]
growth_scaled = scaled[0][3]
hours_scaled = scaled[0][4]
performance_scaled = scaled[0][5]
satisfaction_scaled = scaled[0][6]

# Create model input (MUST match training order)
input_data = pd.DataFrame({
    "Job_Title":[job_title_encoded],
    "Industry":[industry_encoded],
    "Country":[country_encoded],
    "Education_Level":[education_encoded],
    "Years_Experience":[years_scaled],
    "Future_Demand_Score":[future_scaled],
    "Remote_Work_Possibility":[remote_encoded],
    "Average_Salary_USD":[salary_scaled],
    "Required_Skills":[skills_encoded],
    "Automation_Level":[automation_encoded],
    "Job_Growth_2030":[growth_scaled],
    "Work_Hours_Per_Week":[hours_scaled],
    "Company_Size":[company_encoded],
    "AI_Tool_Usage":[ai_tool_encoded],
    "Performance_Score":[performance_scaled],
    "Upskilling_Needed":[upskill_encoded],
    "Job_Satisfaction":[satisfaction_scaled],
    "Hiring_Trend_2026":[hiring_encoded]
})

# Prediction
prediction = model.predict(input_data)[0]

st.header("Predicted AI Replacement Risk")

st.metric(
    label="Predicted AI Replacement Risk",
    value=f"{prediction*100:.1f}%"
)

if prediction < 0.30:
    st.success("✅ Low AI replacement risk")
elif prediction < 0.70:
    st.warning("⚠️ Moderate AI replacement risk")
else:
    st.error("🚨 High AI replacement risk")

st.header("Filtered Dataset")

st.dataframe(filtered_df)

st.header("Distribution of AI Replacement Risk")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(filtered_df["AI_Replacement_Risk"], bins=20)

ax.set_xlabel("AI Replacement Risk")
ax.set_ylabel("Number of Jobs")

st.pyplot(fig)

st.header("Jobs by Industry")

industry_counts = filtered_df["Industry"].value_counts()

fig, ax = plt.subplots(figsize=(10,5))

industry_counts.plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Industry")
ax.set_ylabel("Number of Jobs")

st.pyplot(fig)

st.header("Average Salary vs AI Replacement Risk")

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(
    filtered_df["Average_Salary_USD"],
    filtered_df["AI_Replacement_Risk"]
)

ax.set_xlabel("Average Salary (USD)")
ax.set_ylabel("AI Replacement Risk")

st.pyplot(fig)

# ==================================================
# MODEL MONITORING
# ==================================================

st.header("Model Monitoring Metrics")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Mean Absolute Error (MAE)",
        value="0.2258"
    )

    st.write(
        "Monitors the average prediction error of the deployed Linear Regression model."
    )

missing_values = df.isnull().sum().sum()

with col2:

    st.metric(
        label="Missing Values",
        value=int(missing_values)
    )

    st.write(
        "Monitors data quality by checking for missing values in the dataset."
    )

# ==================================================
# DATA DRIFT ANALYSIS
# ==================================================

st.header("Data Drift Analysis")

# Baseline statistics (training dataset)
baseline_mean = df["AI_Replacement_Risk"].mean()

# Simulated new dataset (assume AI risk has increased slightly)
new_data = df.copy()
new_data["AI_Replacement_Risk"] = new_data["AI_Replacement_Risk"] + 0.05

current_mean = new_data["AI_Replacement_Risk"].mean()

drift = current_mean - baseline_mean

col1, col2, col3 = st.columns(3)

col1.metric("Baseline Mean", f"{baseline_mean:.3f}")
col2.metric("Current Mean", f"{current_mean:.3f}")
col3.metric("Mean Drift", f"{drift:.3f}")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(
    df["AI_Replacement_Risk"],
    bins=20,
    alpha=0.6,
    label="Baseline"
)

ax.hist(
    new_data["AI_Replacement_Risk"],
    bins=20,
    alpha=0.6,
    label="Current"
)

ax.set_xlabel("AI Replacement Risk")
ax.set_ylabel("Frequency")
ax.legend()

st.pyplot(fig)
