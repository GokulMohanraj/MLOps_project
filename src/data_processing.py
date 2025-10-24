import pandas as pd
import numpy as np
import os

class DataProcessor:
    def __init__(self, df: pd.DataFrame, output_path: str = "data/processed_data.csv"):
        self.df = df.copy()
        self.numeric_cols = []
        self.output_path = output_path  # Where to save the cleaned data

    def clean_data(self):
        # Step 1: Replace 'absent' or similar with NaN
        self.df.replace(["absent", "Absent", "ABSENT", ""], np.nan, inplace=True)

        # Step 2: Detect numeric columns automatically
        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors="coerce")

            # If at least one numeric value exists, treat as numeric column
            if not converted.isna().all():
                self.numeric_cols.append(col)
                self.df[col] = converted.fillna(0)  # Replace NaN with 0 (absent → 0 marks)
            else:
                # Keep text columns unchanged (like "Name")
                self.df[col] = self.df[col]

        # Step 3: Create "Total" column by summing all numeric columns
        self.df["Total"] = self.df[self.numeric_cols].sum(axis=1)
        self.numeric_cols.append("Total")

        # Step 4: Handle missing names
        if "Name" in self.df.columns:
            self.df["Name"].fillna("Unknown_Student", inplace=True)

        return self.df

    def assign_grade(self):
        # Exclude "Total" from subject list
        subject_cols = [col for col in self.numeric_cols if col != "Total"]

        # Step 1: Fail if any subject < 35
        fail_mask = (self.df[subject_cols] < 35).any(axis=1)
        self.df.loc[fail_mask, "Grade"] = "Fail"

        # Step 2: Assign grades for passed students
        self.df.loc[~fail_mask & (self.df["Total"] >= 340), "Grade"] = "A"
        self.df.loc[~fail_mask & (self.df["Total"].between(300, 339)), "Grade"] = "B"
        self.df.loc[~fail_mask & (self.df["Total"].between(250, 299)), "Grade"] = "C"
        self.df.loc[~fail_mask & (self.df["Total"] < 250), "Grade"] = "D"

        return self.df

    def save_processed_data(self):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # Save cleaned & graded data
        self.df.to_csv(self.output_path, index=False)
        print(f"✅ Processed data saved successfully at: {self.output_path}")

    def process(self):
        """Full pipeline: clean → grade → save"""
        self.clean_data()
        self.assign_grade()
        self.save_processed_data()
        return self.df
