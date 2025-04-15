# examine_results.py
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend

# Configuration
results_dir = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\model_results'
results_filename = os.path.join(results_dir, 'model_evaluation_results.joblib')
report_dir = 'report'
plot_filename = os.path.join(report_dir, 'feature_importances_plot.png')
N_TOP_FEATURES_TO_SHOW_TEXT = 20
N_TOP_FEATURES_TO_PLOT = 30

print(f"--- Loading Results File: {results_filename} ---")

try:
    results = joblib.load(results_filename)
    print("Results loaded successfully.")
except FileNotFoundError:
    print(f"Error: Results file not found at {results_filename}")
    exit()
except Exception as e:
    print(f"An error occurred loading the results file: {e}")
    exit()

print("\n--- Evaluation Metrics (Test Set) ---")
print(f"Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
print(f"ROC AUC: {results.get('test_roc_auc', 'N/A'):.4f}")
print("\nClassification Report:")
print(results.get('test_classification_report', 'N/A'))

print("\n--- Best Hyperparameters Found --- ")
print(results.get('best_params', 'N/A'))

print(f"\n--- Top {N_TOP_FEATURES_TO_SHOW_TEXT} Selected Features (out of {len(results.get('selected_features', []))}) ---")
selected_features = results.get('selected_features', [])
if selected_features:
    print(selected_features[:N_TOP_FEATURES_TO_SHOW_TEXT])
else:
    print("Selected features list not found in results.")

print(f"\n--- Top {N_TOP_FEATURES_TO_SHOW_TEXT} Feature Importances (from initial fit) ---")
feature_importances_all = results.get('feature_importance_all', {})
if feature_importances_all:
    # Convert back to Series for easier handling
    importances_series = pd.Series(feature_importances_all)
    # Ensure it's sorted
    importances_series = importances_series.sort_values(ascending=False)
    print(importances_series.head(N_TOP_FEATURES_TO_SHOW_TEXT))

    # --- Plotting --- 
    print(f"\n--- Generating Feature Importance Plot (Top {N_TOP_FEATURES_TO_PLOT}) ---")
    top_n_importances = importances_series.head(N_TOP_FEATURES_TO_PLOT)

    plt.figure(figsize=(10, N_TOP_FEATURES_TO_PLOT * 0.35)) # Adjust height based on N
    plt.barh(top_n_importances.index[::-1], top_n_importances.values[::-1], align='center')
    plt.xlabel('XGBoost Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f'Top {N_TOP_FEATURES_TO_PLOT} Feature Importances (Initial Fit)')
    plt.gca().margins(y=0.01) # Add small margins to y-axis
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    try:
        plt.savefig(plot_filename)
        print(f"Plot saved successfully to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close() # Close the plot figure
    # --- End Plotting ---

else:
    print("Feature importances not found in results.")

print("\n--- Examination Complete --- ") 