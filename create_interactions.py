import pandas as pd
import numpy as np
import os
import re # To sanitize column names

# --- Configuration ---
original_data_path = r'C:\\Users\\ortho\\OneDrive\\Documents\\AI\\XGBoost\\predictive_dataset.csv'
split_data_dir = r'C:\\Users\\ortho\\OneDrive\\Documents\\AI\\XGBoost\\split_data'
output_dir = r'C:\\Users\\ortho\\OneDrive\\Documents\\AI\\XGBoost\\interaction_data'
X_train_file = os.path.join(split_data_dir, 'X_train.csv')
X_val_file = os.path.join(split_data_dir, 'X_val.csv')
X_test_file = os.path.join(split_data_dir, 'X_test.csv')

# --- Helper Function to Sanitize Column Names ---
def sanitize_col_name(name):
    # Replace common problematic characters with underscore
    name = re.sub(r'[^a-zA-Z0-9_]+', '_', str(name))
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it doesn't start with a number (less common issue, but safe)
    if name and name[0].isdigit():
        name = '_' + name
    # Handle potential empty names after sanitization
    if not name:
        return "_sanitized_"
    return name

# --- Helper Function to Add Interactions ---
def add_interaction_features(df, numerical_cols, ohe_cols):
    """
    Adds interaction features between numerical and one-hot encoded columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names.
        ohe_cols (list): List of one-hot encoded column names.

    Returns:
        pd.DataFrame: DataFrame with original columns plus interaction features.
                      Returns a copy of the original df if no valid interactions possible.
    """
    print(f"  Processing DataFrame with shape: {df.shape}")
    df_interact = df.copy() # Work on a copy
    interaction_count = 0

    # Store original columns to ensure they are kept
    original_cols = df.columns.tolist()
    new_interaction_cols = [] # Keep track of newly added columns

    # Check if numerical_cols and ohe_cols are valid lists
    if not isinstance(numerical_cols, list) or not isinstance(ohe_cols, list):
        print("  Warning: numerical_cols or ohe_cols are not valid lists. Skipping interaction feature creation.")
        return df_interact # Return the original copy

    # Filter numerical_cols to only include those present in df_interact
    valid_numerical_cols = [col for col in numerical_cols if col in df_interact.columns]
    if len(valid_numerical_cols) < len(numerical_cols):
        print(f"  Warning: Some numerical columns were not found in this DataFrame.")
        # print(f"  Missing numerical columns: {list(set(numerical_cols) - set(valid_numerical_cols))}") # Optional detail
    if not valid_numerical_cols:
        print("  Warning: No valid numerical columns found in this DataFrame. Skipping interaction feature creation.")
        return df_interact # Return the original copy

    # Filter ohe_cols to only include those present in df_interact
    valid_ohe_cols = [col for col in ohe_cols if col in df_interact.columns]
    if len(valid_ohe_cols) < len(ohe_cols):
        print(f"  Warning: Some OHE columns were not found in this DataFrame.")
        # print(f"  Missing OHE columns: {list(set(ohe_cols) - set(valid_ohe_cols))}") # Optional detail
    if not valid_ohe_cols:
        print("  Warning: No valid OHE columns found in this DataFrame. Skipping interaction feature creation.")
        return df_interact # Return the original copy


    print(f"  Generating interactions between {len(valid_numerical_cols)} numerical and {len(valid_ohe_cols)} OHE columns...")

    for num_col in valid_numerical_cols:
        # Check if num_col exists (redundant due to filter, but safe)
        if num_col not in df_interact.columns:
             print(f"  Warning: Numerical column '{num_col}' not found in this DataFrame. Skipping its interactions.")
             continue

        for ohe_col in valid_ohe_cols: # Use the filtered valid_ohe_cols
            if ohe_col not in df_interact.columns:
                # This check is technically redundant now due to pre-filtering valid_ohe_cols,
                # but kept for robustness against potential logic changes.
                print(f"  Warning: OHE column '{ohe_col}' not found in this DataFrame. Skipping interaction with '{num_col}'.")
                continue

            # Create sanitized interaction column name
            interaction_col_name_base = f"{num_col}_x_{ohe_col}"
            interaction_col_name = sanitize_col_name(interaction_col_name_base)

            # Calculate interaction only if the column doesn't already exist
            if interaction_col_name not in df_interact.columns:
                 df_interact[interaction_col_name] = df_interact[num_col] * df_interact[ohe_col]
                 new_interaction_cols.append(interaction_col_name) # Track new col
                 interaction_count += 1
            # else: # Optional: Notify if interaction column already exists (e.g., from a previous run)
            #    print(f"  Info: Interaction column '{interaction_col_name}' already exists. Skipping recalculation.")


    print(f"  Added {interaction_count} new interaction features.")
    print(f"  New shape: {df_interact.shape}") # Shape includes original + new interaction cols

    # Ensure all original columns and new interaction columns are present
    # This step is implicitly handled by adding new columns to df_interact (which started as a copy)
    # and not explicitly selecting a subset based on numerical/ohe lists.
    # We just return the modified df_interact.
    return df_interact

# --- Main Script Logic ---
print("--- Starting Interaction Feature Creation ---")

# Create output directory
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# 1. Identify Original Numerical Columns
print(f"\n--- Identifying Original Numerical Columns from: {original_data_path} ---")
try:
    df_original = pd.read_csv(original_data_path)
    original_numerical_cols = df_original.select_dtypes(include=np.number).columns.tolist()
    # Exclude target-related columns if they were numerical originally and might appear in X_train
    # (assuming 'Survival months' might be related, though it's kept as a feature here based on previous steps)
    print(f"Identified original numerical columns: {original_numerical_cols}")
except FileNotFoundError:
    print(f"Error: Original data file not found at {original_data_path}")
    exit()
except Exception as e:
    print(f"An error occurred loading original data: {e}")
    exit()

# 2. Process Training Data (Define Interactions Here)
print(f"\n--- Processing Training Data: {X_train_file} ---")
try:
    X_train = pd.read_csv(X_train_file)
    # Identify OHE columns based on columns present in X_train
    ohe_cols_train = [col for col in X_train.columns if col not in original_numerical_cols]
    print(f"Identified {len(ohe_cols_train)} OHE columns in X_train.")

    # Add interactions ONLY based on X_train definitions
    X_train_interactions = add_interaction_features(X_train, original_numerical_cols, ohe_cols_train)

    # Save processed training data
    output_path_train = os.path.join(output_dir, 'X_train_interactions.csv')
    X_train_interactions.to_csv(output_path_train, index=False)
    print(f"Saved training data with interactions to: {output_path_train}")

except FileNotFoundError:
    print(f"Error: Training data file not found at {X_train_file}")
    exit()
except Exception as e:
    print(f"An error occurred processing training data: {e}")
    exit()

# 3. Process Validation Data (Apply SAME Interactions)
print(f"\n--- Processing Validation Data: {X_val_file} ---")
try:
    X_val = pd.read_csv(X_val_file)
    # Apply interactions using the SAME numerical and OHE lists derived from X_train
    X_val_interactions = add_interaction_features(X_val, original_numerical_cols, ohe_cols_train)

    # Save processed validation data
    output_path_val = os.path.join(output_dir, 'X_val_interactions.csv')
    X_val_interactions.to_csv(output_path_val, index=False)
    print(f"Saved validation data with interactions to: {output_path_val}")

except FileNotFoundError:
    print(f"Error: Validation data file not found at {X_val_file}")
    exit()
except Exception as e:
    print(f"An error occurred processing validation data: {e}")
    exit()

# 4. Process Test Data (Apply SAME Interactions)
print(f"\n--- Processing Test Data: {X_test_file} ---")
try:
    X_test = pd.read_csv(X_test_file)
    # Apply interactions using the SAME numerical and OHE lists derived from X_train
    X_test_interactions = add_interaction_features(X_test, original_numerical_cols, ohe_cols_train)

    # Save processed test data
    output_path_test = os.path.join(output_dir, 'X_test_interactions.csv')
    X_test_interactions.to_csv(output_path_test, index=False)
    print(f"Saved test data with interactions to: {output_path_test}")

except FileNotFoundError:
    print(f"Error: Test data file not found at {X_test_file}")
    exit()
except Exception as e:
    print(f"An error occurred processing test data: {e}")
    exit()

print("\n--- Interaction Feature Creation Complete ---")
print(f"New datasets saved in: {output_dir}")
print(f"Note: The number of features has increased significantly ({X_train_interactions.shape[1]} columns). Feature selection might be necessary later.") 