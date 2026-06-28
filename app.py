import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page title
st.title("AI Impact on Jobs 2030 Dashboard")

# Load dataset
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

# ------------------------------------
# Interactive Feature 1: Industry Filter
# ------------------------------------
industry = st.selectbox(
    "Select an Industry",
    ["All"] + sorted(df["Industry"].unique().tolist())
)

# Filter by industry
if industry != "All":
    df = df[df["Industry"] == industry]

# ------------------------------------
# Interactive Feature 2: AI Risk Slider
# ------------------------------------
risk = st.slider(
    "Maximum AI Replacement Risk",
    min_value=0,
    max_value=100,
    value=100
)

# Filter by AI Replacement Risk
df = df[df["AI_Replacement_Risk"] <= risk]

# Display dataset
st.header("Dataset Preview")
st.dataframe(df.head())

# ------------------------------------
# Visualization 1
# ------------------------------------
st.header("1. Distribution of AI Replacement Risk")

fig, ax = plt.subplots()
ax.hist(df["AI_Replacement_Risk"], bins=20)
ax.set_xlabel("AI Replacement Risk")
ax.set_ylabel("Frequency")

st.pyplot(fig)

# ------------------------------------
# Visualization 2
# ------------------------------------
st.header("2. Number of Jobs by Industry")

industry_counts = df["Industry"].value_counts()

fig, ax = plt.subplots(figsize=(8,5))
industry_counts.plot(kind="bar", ax=ax)

ax.set_xlabel("Industry")
ax.set_ylabel("Count")

st.pyplot(fig)

# ------------------------------------
# Visualization 3
# ------------------------------------
st.header("3. AI Replacement Risk vs Future Demand Score")

fig, ax = plt.subplots()

ax.scatter(
    df["AI_Replacement_Risk"],
    df["Future_Demand_Score"]
)

ax.set_xlabel("AI Replacement Risk")
ax.set_ylabel("Future Demand Score")

st.pyplot(fig)
