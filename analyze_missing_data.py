# analyze_missing_data.py
import pandas as pd

# Define the path to your dataset
file_path = r'C:\Users\ortho\OneDrive\Documents\AI\XGBoost\predictive_dataset.csv' # Using raw string for Windows path

try:
    # Load the dataset
    df = pd.read_csv(file_path)

    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Combine counts and percentages into a single DataFrame for better readability
    missing_info = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_percentage
    })

    # Filter to show only columns with missing values
    missing_info = missing_info[missing_info['Missing Count'] > 0]

    # Sort by percentage descending
    missing_info = missing_info.sort_values(by='Missing Percentage (%)', ascending=False)

    if missing_info.empty:
        print("No missing values found in the dataset.")
    else:
        print("Missing Data Analysis:")
        print(missing_info)

        # Further analysis suggestion (cannot visualize directly here)
        print("\nTo understand patterns (like MAR, MCAR, MNAR), consider:")
        print("1. Visualizing missingness patterns (e.g., using libraries like missingno).")
        print("2. Checking correlations between missing values in different columns.")
        print("3. Analyzing if missingness in one column relates to values in another.")

except FileNotFoundError:
    print(f"Error: The file was not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}") 