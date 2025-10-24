import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:
    def __init__(self, X, y, model=None, test_size=0.2, random_state=42):
        """
        Initialize ModelTrainer with data and optional parameters.
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.model = model if model else DecisionTreeClassifier(random_state=random_state)
        self.label_encoder = LabelEncoder()

    def train(self):
        """Train the model and print evaluation metrics."""
        # Encode target grades to numbers
        y_encoded = self.label_encoder.fit_transform(self.y)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, y_encoded, test_size=self.test_size, random_state=self.random_state
        )

        # Train the model
        print("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully — Accuracy: {acc}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        return self.model

    def save_model(self, path="models/student_grade_model.joblib"):
        """Save model and label encoder."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder
        }, path)
        print(f"Model saved successfully at: {path}")

    def full_pipeline(self, model_path="models/student_grade_model.joblib"):
        """Run full training and saving pipeline."""
        self.train()
        self.save_model(model_path)
