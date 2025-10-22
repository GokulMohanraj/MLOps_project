import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def clean_data(self):
        # Convert all string columns to lowercase
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].str.lower()

        # Replace 'absent' with NaN
        self.df.replace("absent", np.nan, inplace=True)

        # Automatically detect numeric columns (exclude 'name', 'Grade' and 'expected')
        numeric_cols = [col for col in self.df.columns if col.lower() not in ["name", "expected", "grade"]]

        # Convert to numeric and fill NaN with mean of that column
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col].fillna(self.df[col].mean(), inplace=True)

        # Calculate Total
        self.df["Total"] = self.df[numeric_cols].sum(axis=1)
        self.numeric_cols = numeric_cols + ["Total"]
        return self.df

    def assign_grade(self):
        def grade_func(row):
            if (row[self.numeric_cols[:-1]] < 35).any():
                return "Fail"
            total = row["Total"]
            if total >= 340:
                return "A"
            elif total >= 300:
                return "B"
            elif total >= 250:
                return "C"
            else:
                return "D"
        self.df["Grade"] = self.df.apply(grade_func, axis=1)
        return self.df
