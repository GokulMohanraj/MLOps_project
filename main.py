from src import DataProcessor, ModelTrainer, Predictor
import pandas as pd

# -------------------------------
# STEP 1: Load and preprocess training data
# -------------------------------
df_train = pd.read_csv("data/processed_data.csv")
processor = DataProcessor(df_train)
df_train_clean = processor.clean_data()
df_train_clean = processor.assign_grade()

feature_cols = processor.numeric_cols
X_train = df_train_clean[feature_cols]
y_train = df_train_clean["Grade"]

# -------------------------------
# STEP 2: Train and save model
# -------------------------------
trainer = ModelTrainer(X_train, y_train)
model = trainer.train()
trainer.save_model("models/student_grade_model.pkl")

# -------------------------------
# STEP 3: Load test data and predict
# -------------------------------
df_test = pd.read_csv("data/test_data.csv")
processor_test = DataProcessor(df_test)
df_test_clean = processor_test.clean_data()

X_test = df_test_clean[processor_test.numeric_cols]
predictor = Predictor("models/student_grade_model.pkl")
pred_grades = predictor.predict(X_test)

# -------------------------------
# STEP 4: Compare predictions with expected (first letter)
# -------------------------------
df_test_clean["Predicted_Grade"] = pred_grades
df_test_clean["Expected_First"] = df_test_clean["Expected"].str[0].str.upper()

for i, row in df_test_clean.iterrows():
    predicted_first = row["Predicted_Grade"][0].upper()
    expected_first = row["Expected_First"]
    result = "✅ Correct" if predicted_first == expected_first else "❌ Incorrect"
    print(f"{row['Name']}: Predicted={row['Predicted_Grade']}, Expected={row['Expected']} → {result}")
