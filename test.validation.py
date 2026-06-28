import pandas as pd

# Load dataset
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")

# Test 1: Dataset should not contain missing values
assert df.isnull().sum().sum() == 0, "Dataset contains missing values."

# Test 2: Dataset should not contain duplicate rows
assert df.duplicated().sum() == 0, "Dataset contains duplicate rows."

print("All automated validation tests passed successfully.")
