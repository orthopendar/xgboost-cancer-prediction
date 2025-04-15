# encode_categorical.py
import pandas as pd

# Define paths
input_file_path = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\predictive_dataset.csv'
output_file_path = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\predictive_dataset_encoded.csv'

print(f"--- Loading Data: {input_file_path} ---")
try:
    df = pd.read_csv(input_file_path)
    print("Data loaded successfully.")
    print(f"Original shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file was not found at {input_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# Identify categorical columns (object dtype)
categorical_cols = df.select_dtypes(include='object').columns
print(f"\n--- Identifying Categorical Columns ({len(categorical_cols)}) ---")
print(list(categorical_cols))

# Apply one-hot encoding
print("\n--- Applying One-Hot Encoding ---")
# Setting dummy_na=False to not create columns for NaN (as we found no missing data)
# Setting drop_first=False for now, can be set to True to avoid multicollinearity if needed by the model
df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=False, drop_first=False)

print(f"Shape after encoding: {df_encoded.shape}")
print("First 5 rows of encoded data:")
print(df_encoded.head())

# Save the encoded dataframe
print(f"\n--- Saving Encoded Data to: {output_file_path} ---")
try:
    df_encoded.to_csv(output_file_path, index=False)
    print("Encoded data saved successfully.")
except Exception as e:
    print(f"An error occurred during saving: {e}")

print("\n--- Encoding Complete ---")
print(f"The dataset with one-hot encoded categorical variables is saved as '{output_file_path}'.") 