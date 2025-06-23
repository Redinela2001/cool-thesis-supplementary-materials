Appendix A: Python Code for Inter-Annotator Agreement (Cohen's Kappa & Disagreement Analysis)

# Purpose: This script calculates Cohen's Kappa score to quantify inter-annotator
# agreement (IAA) between two human annotators for categorical data. It also generates
# a confusion matrix for detailed visualization of agreement/disagreement patterns,
# identifies specific instances of disagreement, and visualizes the most frequent
# disagreement pairs. This code is crucial for demonstrating the reliability of
# the manual annotation process.

# Environment and Data Setup for Reproducibility:
# This script was developed with Python 3.11.
# Required Python libraries:
#   - pandas (e.g., 2025.2) for data manipulation
#   - scikit-learn (e.g., 1.5.0) for cohen_kappa_score and confusion_matrix
#   - plotly (e.g., 5.x) for interactive heatmap visualization (requires `pip install plotly`)
#   - matplotlib (e.g., 3.10.0) for static bar chart visualization
#   - seaborn (e.g., 0.13.0) for enhanced visualizations
#
# These libraries can typically be installed via pip:
# `pip install pandas scikit-learn plotly matplotlib seaborn`
#
# Data File: This script assumes a DataFrame `df` is loaded from a prior data
# loading and cleaning script (e.g., the one that prepares your manually annotated
# data). This `df` is expected to contain at least two columns with the labels
# assigned by 'Annotator_1' and 'Annotator_2', and optionally an 'Occurrences'
# column for identifying specific disagreement instances.
# For example, if you have a CSV with annotations:
# df = pd.read_csv("your_annotated_data.csv")
#
# For full reproducibility, all code and data files are available in the supplementary
# online repository at: [Please insert your GitHub repository link here, e.g., https://github.com/yourusername/yourthesisrepo]

# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np # For numerical operations, especially with arrays
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import plotly.figure_factory as ff # For interactive heatmap
import matplotlib.pyplot as plt # For static plotting
import seaborn as sns # For enhanced static plotting

# --- 2. Data Loading and Preparation (Placeholder / Dependency) ---
# This code assumes that the DataFrame 'df' (containing 'Annotator_1' and 'Annotator_2'
# columns, and optionally 'Occurrences') has already been loaded and preprocessed
# by an earlier script or is available in the environment.
#
# Example of how 'df' might be loaded if this script were standalone:
# try:
#     df = pd.read_csv("your_cleaned_annotation_data.csv")
#     df.columns = df.columns.str.strip() # Ensure column names are clean
#     # Ensure columns used for analysis ('Annotator_1', 'Annotator_2') exist
#     if 'Annotator_1' not in df.columns or 'Annotator_2' not in df.columns:
#         raise KeyError("Required columns 'Annotator_1' or 'Annotator_2' not found in DataFrame.")
# except FileNotFoundError:
#     print("Error: Annotation data CSV not found. Please ensure it's loaded or available.")
#     exit()
# except KeyError as e:
#     print(f"Error: Missing expected column for annotation analysis: {e}")
#     exit()

# FOR DEMONSTRATION PURPOSES (REMOVE IN FINAL APPENDIX IF DF IS LOADED ELSEWHERE):
# Create a dummy DataFrame if df is not pre-loaded, just to make the script runnable for testing
# In your actual appendix, df would come from your data loading pipeline.
if 'df' not in locals(): # Check if 'df' variable exists
    print("Warning: 'df' DataFrame not found in current scope. Creating dummy data for demonstration.")
    data = {
        'Occurrences': ['cool instance 1', 'cool instance 2', 'cool instance 3', 'cool instance 4', 'cool instance 5',
                        'cool instance 6', 'cool instance 7', 'cool instance 8', 'cool instance 9', 'cool instance 10'],
        'Annotator_1': ['Basic', 'Emotion', 'Nonliteral', 'Basic', 'Emotion', 'Nonliteral', 'Basic', 'Emotion', 'Basic', 'Nonliteral'],
        'Annotator_2': ['Basic', 'Emotion', 'Basic', 'Basic', 'Nonliteral', 'Nonliteral', 'Emotion', 'Emotion', 'Nonliteral', 'Nonliteral']
    }
    df = pd.DataFrame(data)
    # Ensure columns used are correctly identified
    df['Annotator_1'] = df['Annotator_1'].astype(str)
    df['Annotator_2'] = df['Annotator_2'].astype(str)
# END OF DEMONSTRATION DATA

# Ensure annotator columns are treated as strings to avoid potential type issues
annotator1_labels = df['Annotator_1'].astype(str)
annotator2_labels = df['Annotator_2'].astype(str)

# --- 3. Cohen's Kappa Score Calculation ---
# Calculates Cohen's Kappa, a statistic that measures inter-annotator agreement,
# correcting for chance agreement.
kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)

# Print the calculated Kappa score, formatted to four decimal places.
print(f"Cohen’s Kappa Score: {kappa:.4f}")

# Optional: Interpretation of the Kappa score based on common guidelines.
if kappa < 0:
    interpretation = "Less than chance agreement"
elif 0 <= kappa <= 0.20:
    interpretation = "Slight agreement"
elif 0.21 <= kappa <= 0.40:
    interpretation = "Fair agreement"
elif 0.41 <= kappa <= 0.60:
    interpretation = "Moderate agreement"
elif 0.61 <= kappa <= 0.80:
    interpretation = "Substantial agreement"
elif 0.81 <= kappa <= 1.00:
    interpretation = "Almost perfect agreement"
else:
    interpretation = "Invalid Kappa score (check input values)"
print(f"Interpretation: {interpretation}")

# --- 4. Confusion Matrix Generation and Visualization (Interactive Plotly) ---
# A confusion matrix visually represents the agreement and disagreement between annotators.
# The rows typically represent Annotator 1's labels, and columns represent Annotator 2's.

# Determine all unique labels present in both annotator columns for consistent matrix dimensions.
labels = sorted(list(set(annotator1_labels).union(set(annotator2_labels))))

# Generate the raw confusion matrix.
cm = confusion_matrix(annotator1_labels, annotator2_labels, labels=labels)

# Create an annotated heatmap using Plotly for interactive visualization.
fig_plotly = ff.create_annotated_heatmap(
    z=cm,                  # The confusion matrix data
    x=labels,              # Labels for the x-axis (Annotator 2's labels)
    y=labels,              # Labels for the y-axis (Annotator 1's labels)
    colorscale='Blues',    # Color scheme for the heatmap
    annotation_text=[[str(cell) for cell in row] for row in cm], # Text annotations inside cells
    showscale=True         # Show color scale
)

# Update layout for title and axis labels.
fig_plotly.update_layout(
    title=f'Confusion Matrix (Cohen’s Kappa = {kappa:.4f})', # Title includes Kappa score
    xaxis_title='Annotator 2’s Annotations', # Corrected axis label to Annotator 2
    yaxis_title='Annotator 1’s Annotations'  # Corrected axis label to Annotator 1
)

# Save the interactive plot to an HTML file.
# This HTML file can be opened in a web browser to view the interactive plot.
try:
    fig_plotly.write_html("confusion_matrix_kappa.html")
    print("\nInteractive Confusion Matrix saved to 'confusion_matrix_kappa.html'.")
except Exception as e:
    print(f"Error saving Plotly HTML: {e}")

# In an appendix, you would typically include a static image of this plot (e.g., PNG/JPEG).
# To show it directly in a Python environment, you would use fig_plotly.show().
# This line is commented out for appendix purposes as it's for interactive display.
# fig_plotly.show()

# --- 5. Identifying and Saving Disagreements ---
# Filters the DataFrame to identify and display only those instances where annotators disagreed.

# Filter rows where the labels assigned by Annotator 1 and Annotator 2 are different.
disagreements_df = df[df['Annotator_1'] != df['Annotator_2']].copy() # Use .copy() to avoid SettingWithCopyWarning

# Select relevant columns for displaying disagreements.
# Assuming 'Occurrences' column contains the original text or ID of the annotated item.
disagreements_display = disagreements_df[['Occurrences', 'Annotator_1', 'Annotator_2']]

# Print the total number of disagreement instances.
print(f"\nTotal disagreements identified: {len(disagreements_display)} out of {len(df)} annotations.")

# Save the disagreements to a CSV file.
try:
    disagreements_display.to_csv("annotation_disagreements.csv", index=False)
    print("Disagreement instances saved to 'annotation_disagreements.csv'.")
except Exception as e:
    print(f"Error saving disagreements CSV: {e}")

# In an appendix, you would not include the direct download command.
# This line is commented out for appendix purposes as it's for interactive Colab download.
# files.download("annotation_disagreements.csv")

# --- 6. Analyzing and Visualizing Frequent Disagreements (Static Matplotlib/Seaborn Plot) ---
# This section creates a bar plot to highlight the most common types of disagreements.

# Build a DataFrame from the raw confusion matrix for easier manipulation.
conf_df = pd.DataFrame(cm, index=labels, columns=labels)

# Flatten the confusion matrix into a list of disagreement pair combinations.
# Only include off-diagonal (disagreement) cells with a count greater than 0.
disagreement_pairs = []
for i, row_label in enumerate(labels):
    for j, col_label in enumerate(labels):
        if i != j and cm[i][j] > 0: # Check for disagreement (i != j) and a non-zero count
            disagreement_pairs.append({
                'Annotator_1_label': row_label,
                'Annotator_2_label': col_label,
                'Count': cm[i][j]
            })

# Create a DataFrame from the disagreement pairs and sort by count in descending order.
disagreement_df_sorted = pd.DataFrame(disagreement_pairs).sort_values(by='Count', ascending=False)

# Display the top 10 most frequent disagreement pairs.
print("\nTop 10 Most Frequent Disagreements:")
print(disagreement_df_sorted.head(10))

# Visualize the top 10 most frequent disagreements using a bar plot.
plt.figure(figsize=(10, 6)) # Set figure size for better readability
sns.barplot(
    data=disagreement_df_sorted.head(10), # Plot only the top 10 disagreements
    x='Count',
    # Create a combined label for the y-axis (e.g., 'Basic -> Emotion')
    y=disagreement_df_sorted.head(10)['Annotator_1_label'] + " → " + disagreement_df_sorted.head(10)['Annotator_2_label'],
    palette='Reds_r' # Color palette for the bars (reversed reds)
)
plt.title("Top 10 Most Frequent Disagreements Between Annotators")
plt.xlabel("Count of Disagreements")
plt.ylabel("Annotator 1's Label → Annotator 2's Label") # Clearer y-axis label
plt.tight_layout() # Adjust plot to ensure all elements fit
plt.show() # Display the plot

# Save the sorted disagreement DataFrame to a CSV file.
try:
    disagreement_df_sorted.to_csv("sorted_annotation_disagreements.csv", index=False)
    print("\nSorted disagreement pairs saved to 'sorted_annotation_disagreements.csv'.")
except Exception as e:
    print(f"Error saving sorted disagreements CSV: {e}")

# In an appendix, you would not include the direct download command.
# This line is commented out for appendix purposes as it's for interactive Colab download.
# files.download("sorted_annotation_disagreements.csv")
