# train_evaluate_xgboost.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib # For saving the model and results
import os
import time
import re # Import regex for sanitization

# --- Configuration ---
interaction_data_dir = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\interaction_data'
target_data_dir = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\split_data' # y files are here
output_dir = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\model_results'
model_filename = os.path.join(output_dir, 'final_xgboost_model.json')
results_filename = os.path.join(output_dir, 'model_evaluation_results.joblib')

# Files
X_train_file = os.path.join(interaction_data_dir, 'X_train_interactions.csv')
y_train_file = os.path.join(target_data_dir, 'y_train.csv')
X_val_file = os.path.join(interaction_data_dir, 'X_val_interactions.csv')
y_val_file = os.path.join(target_data_dir, 'y_val.csv')
X_test_file = os.path.join(interaction_data_dir, 'X_test_interactions.csv')
y_test_file = os.path.join(target_data_dir, 'y_test.csv')

# Parameters
N_FEATURES_TO_SELECT = 150 # Select top N features based on initial importance
N_HYPERPARAM_ITER = 50     # Number of iterations for RandomizedSearchCV
CV_FOLDS = 3               # Number of cross-validation folds for tuning
EARLY_STOPPING_ROUNDS = 15
RANDOM_STATE = 42

# --- Helper Function to Sanitize Column Names --- (Copied from create_interactions.py)
def sanitize_col_name(name):
    # Replace common problematic characters with underscore
    name = re.sub(r'[^a-zA-Z0-9_]+', '_', str(name))
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure it doesn't start with a number
    if name and name[0].isdigit():
        name = '_' + name
    # Handle potential empty names after sanitization
    if not name:
        return "_sanitized_"
    return name

# --- Create Output Directory ---
os.makedirs(output_dir, exist_ok=True)
print(f"--- Output directory: {output_dir} ---")

# --- 1. Load Data ---
print("\n--- Loading Data ---")
try:
    X_train_full = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file).iloc[:, 0] # Read first column as Series
    X_val_full = pd.read_csv(X_val_file)
    y_val = pd.read_csv(y_val_file).iloc[:, 0]
    X_test_full = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file).iloc[:, 0]
    print(f"X_train_full shape: {X_train_full.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val_full shape: {X_val_full.shape}")
    print(f"X_test_full shape: {X_test_full.shape}")

    # --- <<< NEW: Sanitize column names AFTER loading >>> ---
    print("\n--- Sanitizing Column Names for XGBoost --- ")
    original_cols = X_train_full.columns.tolist()
    sanitized_cols = [sanitize_col_name(col) for col in original_cols]
    # Check for duplicate sanitized names (can happen if original names only differ by special chars)
    if len(sanitized_cols) != len(set(sanitized_cols)):
        print("Warning: Duplicate column names generated after sanitization. Appending indices.")
        # Simple de-duplication by appending index
        seen = {}
        final_cols = []
        for i, col in enumerate(sanitized_cols):
            if col in seen:
                seen[col] += 1
                final_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                final_cols.append(col)
        sanitized_cols = final_cols

    X_train_full.columns = sanitized_cols
    X_val_full.columns = sanitized_cols # Use the same sanitized names for consistency
    X_test_full.columns = sanitized_cols
    print("Column names sanitized.")
    # print("Example sanitized names:", sanitized_cols[:10]) # Optional check
    # --- <<< End of Sanitization >>> ---

    # --- <<< Add Debugging >>> ---
    print("\n--- Debug: Columns in features data BEFORE sanitization ---")
    print(X_train_full.columns.tolist()[:20]) # Print first 20 columns
    print(f"Is 'Year of diagnosis' in columns? {'Year of diagnosis' in X_train_full.columns}")
    # --- <<< End Debugging >>> ---

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit()
except Exception as e:
    print(f"An error occurred loading data: {e}")
    exit()

# --- 2. Initial Model Training (for Feature Importance) ---
print("\n--- Initial Training for Feature Importance ---")
start_time = time.time()
# Use fewer estimators for faster initial training
initial_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=200, # Reduced for speed
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    random_state=RANDOM_STATE
)

initial_model.fit(
    X_train_full, y_train,
    eval_set=[(X_val_full, y_val)],
    verbose=False # Suppress verbose output for initial fit
)
print(f"Initial training completed in {time.time() - start_time:.2f} seconds.")

# --- 3. Feature Selection ---
print(f"\n--- Selecting Top {N_FEATURES_TO_SELECT} Features ---")
feature_importances = pd.Series(initial_model.feature_importances_, index=X_train_full.columns)
top_features = feature_importances.nlargest(N_FEATURES_TO_SELECT).index.tolist()

# Filter datasets
X_train = X_train_full[top_features]
X_val = X_val_full[top_features]
X_test = X_test_full[top_features]

print(f"Selected {len(top_features)} features.")
print(f"X_train (selected) shape: {X_train.shape}")
print(f"X_val (selected) shape: {X_val.shape}")
print(f"X_test (selected) shape: {X_test.shape}")
# print("Top features selected:", top_features[:10]) # Print a few top features

# --- 4. Hyperparameter Tuning (Randomized Search) ---
print("\n--- Hyperparameter Tuning with RandomizedSearchCV ---")
start_time = time.time()
# Define parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.1, 0.5, 1, 2]
}

# Setup RandomizedSearchCV
# Note: We fit on X_train, CV splits will be used internally.
# Early stopping is harder to integrate directly here, often done in the final fit.
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=RANDOM_STATE
)

random_search = RandomizedSearchCV(
    xgb_clf,
    param_distributions=param_dist,
    n_iter=N_HYPERPARAM_ITER,
    scoring='roc_auc', # Use AUC for evaluation metric
    cv=CV_FOLDS,
    verbose=1, # Show progress
    n_jobs=-1, # Use all available cores
    random_state=RANDOM_STATE
)

random_search.fit(X_train, y_train) # Fit on selected features

print(f"Hyperparameter tuning completed in {time.time() - start_time:.2f} seconds.")
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best AUC score during CV: {random_search.best_score_:.4f}")

best_params = random_search.best_params_

# --- 5. Final Model Training ---
print("\n--- Training Final Model with Best Parameters ---")
start_time = time.time()
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    random_state=RANDOM_STATE,
    **best_params # Unpack best parameters found
)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)], # Use validation set for early stopping
    verbose=False # Can set to True or a number to see progress
)
print(f"Final model training completed in {time.time() - start_time:.2f} seconds.")

# --- 6. Evaluation on Test Set ---
print("\n--- Evaluating Final Model on Test Set ---")
y_pred_proba = final_model.predict_proba(X_test)[:, 1] # Probabilities for the positive class
y_pred = final_model.predict(X_test) # Class predictions

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
class_report = classification_report(y_test, y_pred)

print(f"Test Set Accuracy: {accuracy:.4f}")
print(f"Test Set ROC AUC Score: {roc_auc:.4f}")
print("Test Set Classification Report:")
print(class_report)

# --- 7. Save Model & Results ---
print("\n--- Saving Final Model and Evaluation Results ---")
try:
    # Save model using XGBoost's recommended method
    final_model.save_model(model_filename)
    print(f"Model saved to: {model_filename}")

    # Save results (best params, metrics)
    results = {
        'best_params': best_params,
        'test_accuracy': accuracy,
        'test_roc_auc': roc_auc,
        'test_classification_report': class_report,
        'selected_features': top_features,
        'feature_importance_all': feature_importances.sort_values(ascending=False).to_dict() # Save all importances
    }
    joblib.dump(results, results_filename)
    print(f"Evaluation results saved to: {results_filename}")

except Exception as e:
    print(f"An error occurred saving model/results: {e}")

print("\n--- XGBoost Training and Evaluation Complete ---") 