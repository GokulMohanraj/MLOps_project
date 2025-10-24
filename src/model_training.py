from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class ModelTrainer:
    def __init__(self, df, target_col="Grade"):
        self.df = df.copy()
        self.target_col = target_col
        self.model = DecisionTreeClassifier(random_state=42)
        self.label_encoder = None
        self.feature_cols = None

    def prepare_features(self):
        # Numeric columns
        numeric_cols = [col for col in self.df.columns if col not in ["Name", self.target_col, "Expected"]]
        self.feature_cols = numeric_cols
        return self.df

    def encode_target(self):
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.df[self.target_col])
        return y_encoded

    def train(self):
        self.prepare_features()
        y_encoded = self.encode_target()
        X = self.df[self.feature_cols]
        self.model.fit(X, y_encoded)
        return self.model

    def save_model(self, path="models/student_grade_model.joblib"):
        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_cols": self.feature_cols
        }, path)
        print(f"💾 Model saved at {path}")
