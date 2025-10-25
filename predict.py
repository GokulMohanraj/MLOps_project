import os
import pandas as pd
from src.prediction import Predictor  # or src.predictor depending on your file name

def main():
    print("🚀 Starting Prediction Pipeline...")

    # Paths
    new_data_path = "data/test_data.csv"
    processed_data_path = "data/processed/processed_test_data.csv"
    predicted_save_path = "data/prediction/predicted_data.csv"
    model_path = "models/student_grade_model.joblib"

    # Make sure folders exist
    os.makedirs(os.path.dirname(predicted_save_path), exist_ok=True)

    # Step 1: Load new student data
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"❌ New data not found at: {new_data_path}")
    df_new = pd.read_csv(new_data_path)
    print(f"✅ Loaded {len(df_new)} rows from {new_data_path}")

    # Step 2: Load model and predict
    predictor = Predictor(model_path)
    predictions_df = predictor.predict(df_new,processed_data_path)

    # Step 3: Save predictions
    predictions_df.to_csv(predicted_save_path, index=False)
    print(f"✅ Predictions saved at {predicted_save_path}")

    # Optional: Display first 5 rows
    print(predictions_df.head())

if __name__ == "__main__":
    main()
