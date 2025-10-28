import joblib
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, r2_score, precision_score, recall_score


class ModelTrainer:
    def __init__(self, df, model_class, model_name = None, **model_params):
        self.df = df.copy()
        self.target_col = "Grade"
        self.model = model_class(**model_params)
        self.model_name = model_name or model_class.__name__
        self.label_encoder = None
        self.feature_cols = None

    def prepare_features(self):
        # Numeric columns (exclude Name, Grade, Expected)
        numeric_cols = [col for col in self.df.columns if col not in ["Name", self.target_col, "Expected"]]
        self.feature_cols = numeric_cols
        return self.df

    def encode_target(self):
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.df[self.target_col])
        return y_encoded

    def train(self):
        # Prepare features
        self.prepare_features()
        y_encoded = self.encode_target()
        X = self.df[self.feature_cols]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Dynamically create MLflow run name including max_depth if available
        max_depth = getattr(self.model, "max_depth", None)
        run_name = self.model_name
        if max_depth is not None:
            run_name += f"_max_depth_{max_depth}"

        # Start MLflow experiment tracking
        mlflow.set_experiment("Student_Grade_Prediction")

        with mlflow.start_run(run_name=self.model_name):
            # Train the model
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Evaluate performance
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            r2 = r2_score(y_test, y_pred)
            print(f"✅ Model trained successfully | Accuracy: {acc:.2f}")

            # 🔹 Log parameters & metrics
            mlflow.log_param("model_type", self.model_name)
            mlflow.log_params(self.model.get_params())
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("r2_score", r2)

            # 🔹 Log model to MLflow
            mlflow.sklearn.log_model(self.model, run_name)

        return self.model

    def save_model(self, path=None):
        """
        Save model dynamically using the path parameter.
        If path is None, use default 'models/student_grade_model.joblib'
        """
        if path is None:
            path = f"models/student_grade_model_{self.model_name}.joblib"
        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_cols": self.feature_cols
        }, path)
        print(f"💾 Model saved at {path}")
