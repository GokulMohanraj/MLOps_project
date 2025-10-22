import joblib

class Predictor:
    def __init__(self, model_path):
        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.grade_mapping = saved["grade_mapping"]
        
    def predict(self, X):
        pred_codes = self.model.predict(X)
        return [self.grade_mapping[code] for code in pred_codes]
