import pandas as pd
import numpy as np
import joblib

# Load trained model + grade mapping
saved = joblib.load("models/student_grade_model.pkl")
model = saved["model"]
grade_mapping = saved["grade_mapping"]

# Manually enter student data with expected grades
df = pd.read_csv("data/test_data.csv")

# Clean data

df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df = df.replace("absent", np.nan)

# Convert numeric columns and fill NaN with 0
numeric_cols = [col for col in df.columns if col.lower() not in ["name", "expected"]]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Add Total column
df["Total"] = df[numeric_cols].sum(axis=1)

# Prepare features for prediction
X = df[numeric_cols + ["Total"]]

# Predict numeric codes
pred_codes = model.predict(X)

# Map numeric codes to grade letters
pred_grades = [grade_mapping[code] for code in pred_codes]

# Make Expected column uppercase
df["Expected"] = df["Expected"].str.capitalize()

# Compare with expected grades and print results
for i, student in df.iterrows():
    predicted = pred_grades[i]
    expected = student["Expected"]
    result = "✅ Correct" if predicted == expected else "❌ Incorrect"
    print(f"{student['Name']}: Predicted={predicted}, Expected={expected} → {result}")
