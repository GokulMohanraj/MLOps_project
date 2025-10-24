import pandas as pd
import os
from src import DataProcessor, ModelTrainer


# Step 1: Load raw student marks data
raw_data_path = "data/student_marks.csv"
df = pd.read_csv(raw_data_path)

# Step 2: Clean data and assign grades
processor = DataProcessor(df)
df_clean = processor.clean_data()
df_with_grades = processor.assign_grade()  # Generates 'Grade' and 'HasFailedSubject'

# Optional: Inspect cleaned data
print("✅ Sample cleaned data with grades:")
print(df_with_grades.head())

# Step 3: Train ML model
trainer = ModelTrainer(df_with_grades)
trainer.train()

# Step 4: Save trained model with dynamic path
model_path = "models/student_grade_model_v1.joblib"  # You can change this dynamically
os.makedirs(os.path.dirname(model_path), exist_ok=True)
trainer.save_model(path=model_path)

print("✅ Model training completed and saved successfully!")
