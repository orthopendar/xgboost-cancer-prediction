# split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define paths and parameters
encoded_file_path = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\predictive_dataset_encoded.csv'
output_dir = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\split_data'
# --- Target Variable Definition ---
target_column_base = 'SEER cause-specific death classification'
target_column_positive_class = f'{target_column_base}_Dead (attributable to this cancer dx)'
# --- End Target Variable Definition ---
test_size = 0.20  # 20% for the test set
validation_size = 0.20 # 20% of the remaining 80% for validation (0.20 * 0.80 = 0.16)
random_state = 42 # For reproducibility

print(f"--- Loading Encoded Data: {encoded_file_path} ---")
try:
    df = pd.read_csv(encoded_file_path)
    print("Encoded data loaded successfully.")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: The file was not found at {encoded_file_path}")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"\n--- Output directory: {output_dir} ---")

# --- Target Variable Check --- (Check for the specific positive class column)
if target_column_positive_class not in df.columns:
    print(f"Error: Target column '{target_column_positive_class}' not found in the dataset.")
    similar_cols = [col for col in df.columns if target_column_base in col]
    if similar_cols:
        print("Found related columns from the original target:")
        for scol in similar_cols:
            print(f"  - {scol}")
    exit()
else:
    print(f"Target column identified: '{target_column_positive_class}'")

# --- <<< MODIFIED: Identify ALL columns derived from the original target >>> ---
columns_to_drop_from_features = [col for col in df.columns if col.startswith(target_column_base)]
print(f"\n--- Columns to remove from features (derived from original target): ---")
print(columns_to_drop_from_features)
# --- <<< End Modification >>> ---

print(f"\n--- Separating Features (X) and Target (y) --- ")
# --- <<< MODIFIED: Drop ALL related target columns from X >>> ---
X = df.drop(columns=columns_to_drop_from_features)
# --- <<< End Modification >>> ---
y = df[target_column_positive_class] # Target is only the positive class column
print(f"Features shape (X) after dropping target-related columns: {X.shape}")
print(f"Target shape (y): {y.shape}")
print(f"Target value counts:\n{y.value_counts(normalize=True)}") # Check class distribution

# --- Splitting Data ---
print("\n--- Splitting into Training/Validation and Test Sets ---")
# First split: Separate test set (stratified)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=y # Ensure class distribution is similar in test set
)
print(f"Temp (Train+Val) X shape: {X_temp.shape}")
print(f"Test X shape: {X_test.shape}")

print("\n--- Splitting Temp into Training and Validation Sets ---")
# Second split: Separate training and validation sets from the temporary set (stratified)
# Adjust validation size relative to the temporary set size
relative_val_size = validation_size / (1 - test_size)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=relative_val_size,
    random_state=random_state,
    stratify=y_temp # Ensure class distribution is similar in validation set
)

print(f"\n--- Final Split Shapes ---")
print(f"Train X shape: {X_train.shape}, Train y shape: {y_train.shape}")
print(f"Validation X shape: {X_val.shape}, Validation y shape: {y_val.shape}")
print(f"Test X shape: {X_test.shape}, Test y shape: {y_test.shape}")

# Verify stratification (optional check)
print("\nTarget distribution in splits:")
print(f"Train: \n{y_train.value_counts(normalize=True)}")
print(f"Validation: \n{y_val.value_counts(normalize=True)}")
print(f"Test: \n{y_test.value_counts(normalize=True)}")

# --- Saving Split Data ---
print("\n--- Saving Split Data Files --- ")
try:
    # --- <<< Ensure correct column names are used when saving >>> ---
    y_train_df = y_train.to_frame(name=target_column_positive_class)
    y_val_df = y_val.to_frame(name=target_column_positive_class)
    y_test_df = y_test.to_frame(name=target_column_positive_class)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    y_train_df.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    y_val_df.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_test_df.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    # --- <<< End Modification >>> ---
    print("Split data saved successfully.")
except Exception as e:
    print(f"An error occurred saving split data: {e}")

print("\n--- Data Splitting Complete --- ") 