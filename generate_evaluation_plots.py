import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, RocCurveDisplay,
    precision_recall_curve, average_precision_score, PrecisionRecallDisplay,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibrationDisplay
import shap
import re

# --- Configuration ---
model_results_dir = 'model_results'
split_data_dir = 'split_data'
interaction_data_dir = 'interaction_data'
report_dir = 'report'

model_filename = os.path.join(model_results_dir, 'final_xgboost_model.json')
results_filename = os.path.join(model_results_dir, 'model_evaluation_results.joblib')
test_features_filename = os.path.join(interaction_data_dir, 'X_test_interactions.csv')
test_labels_filename = os.path.join(split_data_dir, 'y_test.csv')

# Ensure report directory exists
os.makedirs(report_dir, exist_ok=True)
print(f"Report directory: {report_dir}")

# --- Helper Function to Sanitize Column Names ---
# (Copied from other scripts for self-containment)
def sanitize_col_name(name):
    name = re.sub(r'[^a-zA-Z0-9_]+', '_', str(name))
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if name and name[0].isdigit():
        name = '_' + name
    if not name:
        return "_sanitized_"
    return name

# --- Load Data and Model ---
print("--- Loading Data and Model ---")
try:
    print(f"Loading model from {model_filename}...")
    model = xgb.XGBClassifier()
    model.load_model(model_filename)
    print("Model loaded.")

    print(f"Loading evaluation results from {results_filename}...")
    results = joblib.load(results_filename)
    selected_features_original = results['selected_features'] # Load original names
    print(f"Loaded {len(selected_features_original)} selected features (original names).")

    print(f"Loading test features from {test_features_filename}...")
    X_test_full = pd.read_csv(test_features_filename, low_memory=False)
    print(f"Test features loaded. Shape: {X_test_full.shape}")

    print(f"Loading test labels from {test_labels_filename}...")
    y_test = pd.read_csv(test_labels_filename)
    # Ensure y_test is a Series and get the actual target column name
    if y_test.shape[1] != 1:
         raise ValueError(f"Expected y_test file to have 1 column, found {y_test.shape[1]}")
    target_col_name = y_test.columns[0]
    y_test = y_test[target_col_name]
    print(f"Test labels loaded. Shape: {y_test.shape}")
    print(f"Target column: {target_col_name}")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()


# --- Prepare Test Data ---
print("\n--- Preparing Test Data ---")

# Apply imputation (using median/mode from test set - ideally use training set values)
# Note: This is simplified; for robustness, imputation learned from training data should be used.
print("Applying imputation (using median/mode from test set - simplified approach)...")
imputation_applied = False
for col in selected_features_original: # Impute based on original selected names before sanitization
    if col in X_test_full.columns:
        if X_test_full[col].isnull().any():
             imputation_applied = True
             if pd.api.types.is_numeric_dtype(X_test_full[col]):
                 median_val = X_test_full[col].median()
                 X_test_full[col] = X_test_full[col].fillna(median_val)
             else:
                 if not X_test_full[col].mode().empty:
                     mode_val = X_test_full[col].mode()[0]
                     X_test_full[col] = X_test_full[col].fillna(mode_val)
                 else:
                     print(f"Warning: Could not determine mode for imputation in column '{col}'. Filling NaNs with 'Unknown'.")
                     X_test_full[col] = X_test_full[col].fillna('Unknown') # Fallback
    else:
         # Only warn if a selected feature is genuinely missing *before* sanitization
         print(f"Warning: Selected feature '{col}' (original name) not found in loaded test data columns for imputation.")
if not imputation_applied:
    print("No missing values found in selected features for imputation.")


# Sanitize column names
print("Sanitizing column names...")
original_columns_list = X_test_full.columns.tolist()
X_test_full.columns = [sanitize_col_name(col) for col in original_columns_list]
# Sanitize the list of selected features
selected_features_sanitized = [sanitize_col_name(col) for col in selected_features_original]

# Select features based on sanitized names
print(f"Selecting {len(selected_features_sanitized)} features...")
missing_features = [f for f in selected_features_sanitized if f not in X_test_full.columns]
if missing_features:
    print(f"Error: The following selected features (sanitized) are missing from the test data after sanitization: {missing_features}")
    print("\nAvailable sanitized columns in test data:")
    print(sorted(X_test_full.columns))
    exit()

X_test = X_test_full[selected_features_sanitized]
print(f"Test data prepared. Shape: {X_test.shape}")

# --- Generate Predictions ---
print("\n--- Generating Predictions ---")
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of positive class (class 1)
y_pred_class = model.predict(X_test) # Predicted class labels (0 or 1)
print("Predictions generated.")

# --- Generate Plots ---
print("\n--- Generating Evaluation Plots ---")

# 1. Target Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_test)
plt.title('Distribution of Target Variable in Test Set')
plt.xlabel(f'{target_col_name} (0: Alive/Other, 1: Cancer Death)')
plt.ylabel('Count')
plt.tight_layout()
plot_path = os.path.join(report_dir, 'target_distribution.png')
plt.savefig(plot_path)
plt.close()
print(f"Target distribution plot saved to {plot_path}")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='XGBoost')
roc_display.plot(ax=plt.gca())
plt.title('ROC Curve (Test Set)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plot_path = os.path.join(report_dir, 'roc_curve.png')
plt.savefig(plot_path)
plt.close()
print(f"ROC curve plot saved to {plot_path}")

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision, estimator_name='XGBoost')
pr_display.plot(ax=plt.gca())
plt.title('Precision-Recall Curve (Test Set)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plot_path = os.path.join(report_dir, 'pr_curve.png')
plt.savefig(plot_path)
plt.close()
print(f"PR curve plot saved to {plot_path}")

# 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(7, 5))
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False) # Use Blues colormap
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plot_path = os.path.join(report_dir, 'confusion_matrix.png')
plt.savefig(plot_path)
plt.close()
print(f"Confusion matrix plot saved to {plot_path}")

# 5. Feature Importance Plot (using 'gain')
try:
    importance_type = 'gain'
    feature_importances = model.get_booster().get_score(importance_type=importance_type)
    if not feature_importances:
         print(f"Warning: Could not retrieve feature importance using type '{importance_type}'. Trying 'weight'.")
         importance_type = 'weight'
         feature_importances = model.get_booster().get_score(importance_type=importance_type)

    if feature_importances:
        sorted_idx = np.argsort(list(feature_importances.values()))
        top_n = 30 # Show top N features
        sorted_idx = sorted_idx[-top_n:]

        plt.figure(figsize=(10, max(6, top_n * 0.3))) # Adjust height based on N
        plt.barh([list(feature_importances.keys())[i] for i in sorted_idx],
                 [list(feature_importances.values())[i] for i in sorted_idx])
        plt.xlabel(f"XGBoost Feature Importance ({importance_type.capitalize()})")
        plt.ylabel("Feature")
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plot_path = os.path.join(report_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Feature importance plot saved to {plot_path}")
    else:
        print("Could not retrieve any feature importances from the model.")

except Exception as e:
    print(f"Error generating feature importance plot: {e}")


# 6. SHAP Summary Plot
print("Calculating SHAP values (this may take a moment)...")
try:
    # Use TreeExplainer for tree-based models like XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test) # Use the prepared test set

    # Create summary plot
    plt.figure() # Create a new figure context for SHAP
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    # Adjust layout after SHAP plot potentially modifies it
    plt.tight_layout()
    plot_path = os.path.join(report_dir, 'shap_summary.png')
    plt.savefig(plot_path, bbox_inches='tight') # Use bbox_inches for SHAP plots
    plt.close()
    print(f"SHAP summary plot saved to {plot_path}")

except Exception as e:
    print(f"Error generating SHAP plot: {e}")
    # If SHAP fails, especially on Windows with XGBoost versions, mention potential issues
    if "TreeExplainer" in str(e) or "XGBoost" in str(e):
        print("Note: SHAP calculations can sometimes have issues with specific XGBoost versions or environments.")


# 7. Calibration Plot
print("Generating Calibration Plot...")
try:
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')

    plt.figure(figsize=(8, 6))
    disp = CalibrationDisplay(prob_true, prob_pred, y_pred_proba)
    disp.plot(ax=plt.gca(), name='XGBoost')
    plt.title('Calibration Plot (Reliability Curve)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(report_dir, 'calibration_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Calibration plot saved to {plot_path}")

except Exception as e:
     print(f"Error generating calibration plot: {e}")


print("\n--- Plot Generation Complete ---") 