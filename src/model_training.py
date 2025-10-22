from sklearn.tree import DecisionTreeClassifier
import joblib

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = DecisionTreeClassifier()
        self.grade_mapping = {i: g for i, g in enumerate(sorted(y.unique()))}
    
    def train(self):
        self.model.fit(self.X, self.y.map(lambda x: list(self.grade_mapping.values()).index(x)))
        return self.model
    
    def save_model(self, path):
        joblib.dump({"model": self.model, "grade_mapping": self.grade_mapping}, path)
