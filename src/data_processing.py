import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = None

    def clean_data(self):
        # Fill missing student names
        self.df["Name"] = self.df["Name"].fillna("Unknown_Student")

        # Automatically detect numeric columns (exclude Name, Grade, Expected)
        numeric_cols = [col for col in self.df.columns if col not in ["Name", "Grade", "Expected"]]

        # Convert numeric columns to int and fill NaN with 0
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)

        # Calculate Total
        self.df["Total"] = self.df[numeric_cols].sum(axis=1)

        self.numeric_cols = numeric_cols + ["Total"]
        return self.df

    def assign_grade(self):
        # Initialize Grade column with default "D"
        self.df["Grade"] = "D"

        # Fail if any subject < 35
        self.df.loc[(self.df[self.numeric_cols[:-1]] < 35).any(axis=1), "Grade"] = "Fail"

        # Assign B, C, A based on Total, skip already Fail
        # C grade: 250 <= Total < 300
        self.df.loc[self.df["Total"].between(250, 299) & (self.df["Grade"] != "Fail"), "Grade"] = "C"
        # B grade: 300 <= Total < 340
        self.df.loc[self.df["Total"].between(300, 339) & (self.df["Grade"] != "Fail"), "Grade"] = "B"
        self.df.loc[(self.df["Total"] >= 340) & (self.df["Grade"] != "Fail"), "Grade"] = "A"

        # Add HasFailedSubject feature
        numeric_cols = [col for col in self.df.columns if col not in ["Name", "Grade", "Expected"]]
        self.df["HasFailedSubject"] = (self.df[numeric_cols[:-1]] < 35).any(axis=1).astype(int)

        return self.df
