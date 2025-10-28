import os
import pandas as pd
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def main():
    print("🚀 Starting Student Grade Prediction Pipeline...")

    # Define paths
    raw_data_path = "data/raw/raw_data.csv"
    model_save_path = None  # Use default path in save_model method

    # Step 1: Load raw data
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"❌ Raw data not found at: {raw_data_path}")
    print("📂 Loading raw data...")
    df_raw = pd.read_csv(raw_data_path)
    print(f"✅ Loaded {len(df_raw)} rows from {raw_data_path}")

    # Step 2: Process data
    print("🧹 Cleaning and processing data...")
    processor = DataProcessor(df_raw, save_path="data/processed/processed_data.csv")
    processed_df = processor.run_pipeline()

     # Step 4: Train the model
    print("🤖 Training model...")
    models = [
        (LogisticRegression, {"max_iter": 750}),
        (RandomForestClassifier, {"n_estimators": 100, "max_depth": 10}),
        (DecisionTreeClassifier, {"max_depth": 5})  
    ]
    for model_class, params in models:
        trainer = ModelTrainer(processed_df, model_class, **params)
        trainer.train()
        trainer.save_model()
    print(f"✅ Model saved successfully at: {model_save_path}")

    print("🎉 Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
