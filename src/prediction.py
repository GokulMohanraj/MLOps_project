import pandas as pd
import joblib
import os
from .data_processing import DataProcessor

class Predictor:
    def __init__(self, model_path=None, processed_data_path=None):
        """
        Initialize Predictor with a model path.
        If no path provided, uses default models/student_grade_model_DecisionTreeClassifier.joblib
        """
        if model_path is None:
            model_path = "models/student_grade_model_DecisionTreeClassifier.joblib"
        self.model_path = model_path
        self.processed_data_path = processed_data_path
        self.model = None
        self.label_encoder = None
        self.feature_cols = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        saved = joblib.load(self.model_path)
        self.model = saved["model"]
        self.label_encoder = saved["label_encoder"]
        self.feature_cols = saved["feature_cols"]
        print(f"✅ Model loaded successfully from {self.model_path}")

    def predict(self, df: pd.DataFrame, processed_data_path):
        # Step 1: Clean new data
        processor = DataProcessor(df, save_path=processed_data_path)
        df_clean = processor.clean_data()

        # Step 2: Create HasFailedSubject feature
        numeric_cols = [col for col in df_clean.columns if col not in ["Name", "Grade", "Expected"]]
        df_clean["HasFailedSubject"] = (df_clean[numeric_cols[:-1]] < 35).any(axis=1).astype(int)

        # Step 3: Predict grades
        X_new = df_clean[self.feature_cols]
        y_pred_encoded = self.model.predict(X_new)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        df_clean["Predicted_Grade"] = y_pred

        return df_clean

    def save_predictions(self, df: pd.DataFrame, output_path="data/predicted_grades.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"💾 Predictions saved at {output_path}")
