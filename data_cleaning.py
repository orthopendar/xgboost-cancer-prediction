# data_cleaning.py
import pandas as pd
import numpy as np

# Define the path to your dataset
file_path = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\predictive_dataset.csv' # Using raw string

print(f"--- Loading Data: {file_path} ---")
try:
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

print("\n--- 1. Schema Validation & Tidiness Check ---")
print("Dataset Info (Column Names, Non-Null Counts, Data Types):")
df.info()

print("\nFirst 5 Rows (Visual Tidiness Check):")
print(df.head())

print("\n--- 2. Inconsistencies, Errors & Initial Outlier Check ---")
print("Descriptive Statistics (Numerical Columns):")
print(df.describe(include=np.number))

print("\nDescriptive Statistics (Object/Categorical Columns):")
# Include 'datetime' types if any, otherwise pandas might exclude them
print(df.describe(include=['object', 'category', 'datetime64[ns]']))

# Check unique values for object columns with few unique values
print("\nUnique Values for Potential Categorical Columns (if < 50 unique values):")
object_cols = df.select_dtypes(include='object').columns
for col in object_cols:
    unique_count = df[col].nunique()
    if unique_count < 50:
        print(f"  - {col} ({unique_count} unique): {df[col].unique().tolist()}")
    else:
        print(f"  - {col} ({unique_count} unique): Too many to list here.")


print("\n--- 3. Detailed Outlier Detection (IQR Method) ---")
numerical_cols = df.select_dtypes(include=np.number).columns
outliers_found = False
print("Checking numerical columns for outliers (values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR):")

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter for outliers
    outlier_condition = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_count = df[col][outlier_condition].count()

    if outlier_count > 0:
        outliers_found = True
        print(f"  - Potential outliers found in '{col}': {outlier_count} points")
        # Optional: List some outlier values
        # print(f"    Examples: {df[col][outlier_condition].head().tolist()}")
    else:
         print(f"  - No potential outliers detected in '{col}' based on IQR.")


if not outliers_found:
    print("No potential outliers detected in any numerical column based on the IQR method.")

print("\n--- Data Cleaning Summary ---")
print("1. Schema & Tidiness: Review the df.info() and df.head() output above.")
print("2. Inconsistencies/Errors: Review descriptive statistics and unique values. Look for unexpected ranges, counts, or categories.")
print("3. Outliers: Review the IQR outlier detection results. Decide on handling strategy (remove, transform, keep) if outliers were found.")
print("4. Next Steps: Based on this analysis, decide on specific cleaning actions (e.g., correcting formats, handling outliers).") 