import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = []

    def clean_data(self):
        # Step 1: Replace 'absent' or similar values with NaN
        self.df.replace(["absent", "Absent", "ABSENT", ""], np.nan, inplace=True)

        # Step 2: Detect numeric columns automatically
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors='coerce')

            # If at least one value could be converted → it's a numeric column
            if not converted.isna().all():
                self.numeric_cols.append(col)
                # Replace NaN (absent students) with 0 marks
                self.df[col] = converted.fillna(0)
            else:
                # Keep text columns as they are (e.g., Name)
                self.df[col] = self.df[col]

        # Step 3: Create 'Total' column by summing numeric columns
        self.df["Total"] = self.df[self.numeric_cols].sum(axis=1)
        self.numeric_cols.append("Total")

        # Step 4: Handle missing names
        if "Name" in self.df.columns:
            self.df["Name"].fillna("Unknown_Student", inplace=True)

        return self.df

    def assign_grade(self):
        # Exclude 'Total' column when checking individual subject marks
        subject_cols = [col for col in self.numeric_cols if col != "Total"]

        # Step 1: Fail if any subject score is below 35
        fail_mask = (self.df[subject_cols] < 35).any(axis=1)
        self.df.loc[fail_mask, "Grade"] = "Fail"

        # Step 2: Grade for passed students based on total marks
        self.df.loc[~fail_mask & (self.df["Total"] >= 340), "Grade"] = "A"
        self.df.loc[~fail_mask & (self.df["Total"].between(300, 339)), "Grade"] = "B"
        self.df.loc[~fail_mask & (self.df["Total"].between(250, 299)), "Grade"] = "C"
        self.df.loc[~fail_mask & (self.df["Total"] < 250), "Grade"] = "D"

        return self.df
