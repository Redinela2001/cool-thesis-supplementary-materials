# Appendix C: Python Code for Chi-square Tests and Association Plots (Manually Annotated Data)

# Purpose: This script performs Chi-square tests of independence and generates
# association plots for three key relationships in the manually annotated 'cool' dataset:
# 1. Genre vs. Interpretation (horizontal bar chart style)
# 2. Decade vs. Interpretation (horizontal bar chart style)
# 3. Syntax vs. Interpretation (grouped bar chart style, matching main thesis figure)
# These analyses quantify the strength and visualize the patterns of association
# between these categorical variables, based on observed and expected frequencies,
# and standardized Pearson residuals.

# Environment and Data Setup for Reproducibility:
# This script was developed with Python 3.11.
# Required Python libraries:
#   - pandas (e.g., 2025.2) for data manipulation
#   - numpy (e.g., 2.0.2) for numerical operations
#   - matplotlib (e.g., 3.10.0) for plotting
#   - seaborn (e.g., 0.13.0) for enhanced visualizations
#   - scipy (e.g., 1.13.0) for chi2_contingency statistical test
#   - openpyxl (required by pandas to read .xlsx files, e.g., 3.1.2)
#
# These libraries can typically be installed via pip:
# `pip install pandas numpy matplotlib seaborn scipy openpyxl`
#
# Data File: This script requires an Excel file (.xlsx) containing the 'genre',
# 'decade', 'syntax', and 'interpretation' columns. The `file_path` variable should be updated
# to point to the actual location of your data file.
#
# For full reproducibility, all code and data files are available in the supplementary
# online repository at: [Please insert your GitHub repository link here, e.g., https://github.com/yourusername/yourthesisrepo]

# --- 1. Import Necessary Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency # For Chi-square test
import unicodedata # For robust string cleaning

# --- 2. Data Loading and Cleaning ---
# Specify the path to your Excel file containing the data.
# IMPORTANT: Replace "your_data.xlsx" with the actual path to your file.
file_path = "your_data.xlsx" # e.g., "data/cool_annotations_final.xlsx"

try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: Data file not found at '{file_path}'.")
    print("Please ensure the file path is correct and the file exists.")
    exit() # Exit the script if the data file cannot be found

# Clean column names by stripping whitespace and converting to lowercase for consistency.
df.columns = df.columns.str.strip().str.lower()

# Verify that all necessary columns exist.
required_columns = ['genre', 'decade', 'syntax', 'interpretation']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Your file must contain columns named {required_columns}. "
                     "Please check your Excel file's column headers.")

# Define a robust string cleaning function for categorical values.
# This function handles Unicode normalization and removes accents,
# ensuring consistency in string data before further processing.
def clean_string(s):
    """
    Normalizes Unicode characters, removes accents, and strips leading/trailing whitespace.
    Args:
        s (str): The input string to clean.
    Returns:
        str: The cleaned string.
    """
    s = str(s) # Ensure input is treated as a string
    s = unicodedata.normalize("NFKD", s)  # Normalize Unicode characters (e.g., 'é' to 'e')
    s = s.encode("ascii", "ignore").decode("utf-8")  # Remove non-ASCII characters (accents)
    s = s.strip()  # Remove leading/trailing whitespace
    return s

# Apply robust cleaning and standardization to values within the relevant columns.
# Values are converted to string, cleaned, then lowercased for filtering.
df['genre'] = df['genre'].apply(clean_string).str.lower()
df['decade'] = df['decade'].apply(clean_string).str.lower()
df['syntax'] = df['syntax'].apply(clean_string).str.lower()
df['interpretation'] = df['interpretation'].apply(clean_string).str.lower()

# Define valid interpretation categories to filter out any unexpected values
valid_interpretation_cats = ['basic', 'emotion', 'nonliteral']
df = df[df['interpretation'].isin(valid_interpretation_cats)].copy() # Filter and create a copy

# Capitalize the categories for consistent presentation in tables and plots (e.g., "Basic", "Fiction", "1990s").
# This is applied AFTER filtering to ensure only valid categories are capitalized.
df['genre'] = df['genre'].str.capitalize()
df['decade'] = df['decade'].str.capitalize()
df['syntax'] = df['syntax'].str.capitalize()
df['interpretation'] = df['interpretation'].str.capitalize()

# Drop rows with missing values in the relevant columns for analysis (after cleaning).
df_clean = df.dropna(subset=required_columns).copy() # Use .copy() to avoid SettingWithCopyWarning


# --- Helper Function for Generic Chi-square Test and Horizontal Association Plot ---
def analyze_and_plot_association(data, var1_col, var2_col, plot_title_prefix=""):
    """
    Performs Chi-square test, calculates Cramér's V and Pearson residuals,
    and generates a horizontal bar chart association plot for two given categorical columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        var1_col (str): Name of the first categorical column (for rows in contingency table).
        var2_col (str): Name of the second categorical column (for columns in contingency table).
        plot_title_prefix (str): Prefix for the plot title (e.g., "Genre × Interpretation").
    """
    print(f"\n--- Analyzing: {var1_col.capitalize()} vs. {var2_col.capitalize()} ---")

    # Create contingency table
    contingency_table = pd.crosstab(data[var1_col], data[var2_col])
    print(f"\n🔢 Observed Frequencies ({var1_col.capitalize()} × {var2_col.capitalize()}):")
    print(contingency_table)

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Convert expected frequencies to a DataFrame
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)

    # Calculate Standardized Pearson Residuals
    residuals = (contingency_table - expected_df) / np.sqrt(expected_df)

    # Cramér's V calculation (Effect Size)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    cramers_v = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0 # Avoid division by zero

    # Output Results
    print(f"\n📊 Chi-square Test Results ({var1_col.capitalize()} × {var2_col.capitalize()}):")
    print("------------------------------------------")
    print(f"Chi² Statistic      : {chi2:.4f}")
    print(f"Degrees of Freedom  : {dof}")
    print(f"P-value             : {p:.4e}")
    print(f"Cramér's V          : {cramers_v:.4f}")

    print(f"\n📈 Expected Frequencies ({var1_col.capitalize()} × {var2_col.capitalize()}) (Rounded to 2 decimal places):")
    print(expected_df.round(2))

    print(f"\n🔥 Standardized Pearson Residuals ({var1_col.capitalize()} × {var2_col.capitalize()}) (Rounded to 2 decimal places):")
    print(residuals.round(2))

    # Visualization (Association Plot)
    res_long = residuals.reset_index().melt(id_vars=var1_col, var_name=var2_col.capitalize(), value_name='Residual')
    res_long.rename(columns={var1_col: var1_col.capitalize()}, inplace=True)
    res_long['Label'] = res_long['Residual'].round(2).astype(str)

    vmax = np.ceil(np.abs(res_long['Residual']).max())
    norm = plt.Normalize(-vmax, vmax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])

    plt.figure(figsize=(12, 8)) # Adjust figure size as needed
    sns.set_style("whitegrid") # Optional: Set plot style

    # Create the horizontal bars
    bars = plt.barh(
        y=[f"{s} - {m}" for s, m in zip(res_long[var1_col.capitalize()], res_long[var2_col.capitalize()])],
        width=res_long['Residual'],
        color=plt.cm.coolwarm(norm(res_long['Residual'].values)),
        edgecolor='black'
    )

    # Add residual labels next to the bars
    for bar, label_text in zip(bars, res_long['Label']):
        x_pos = bar.get_width()
        y_pos = bar.get_y() + bar.get_height()/2
        ha = 'left' if x_pos >= 0 else 'right'
        plt.text(x_pos + (0.2 if x_pos >= 0 else -0.2), y_pos, label_text,
                 ha=ha, va='center', fontsize=9)

    # Formatting the plot
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f'Association Plot: {plot_title_prefix} (Standardized Residuals)', fontsize=16)
    plt.xlabel('Standardized Pearson Residuals', fontsize=12)
    plt.ylabel(f'{var1_col.capitalize()} - {var2_col.capitalize()} Pair', fontsize=12)

    # Add colorbar to show the residual scale
    cbar = plt.colorbar(sm, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('Pearson Residuals', rotation=270, labelpad=15, fontsize=12)

    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show() # Display the plot

    # --- Optional: Save Tables as Files ---
    # contingency_table.to_csv(f"observed_frequencies_{var1_col}_{var2_col}.csv")
    # expected_df.to_csv(f"expected_frequencies_{var1_col}_{var2_col}.csv")
    # residuals.to_csv(f"standardized_residuals_{var1_col}_{var2_col}.csv")


# --- Helper Function for Syntax vs Interpretation Plot (Specific Grouped Bar Chart Style) ---
def plot_syntax_vs_interpretation_grouped_bars(data):
    """
    Performs Chi-square test and generates a grouped bar chart association plot
    for Syntax vs. Interpretation, including observed counts as annotations.
    This plot matches the specific style used in the main thesis body.

    Args:
        data (pd.DataFrame): The input DataFrame containing 'syntax' and 'interpretation' columns.
    """
    print("\n--- Analyzing: Syntax vs. Interpretation ---")

    # Create contingency table (Interpretation as rows, Syntax as columns as per original request)
    contingency_table = pd.crosstab(data['interpretation'], data['syntax'])
    print("\n🔢 Observed Frequencies (Interpretation × Syntax):")
    print(contingency_table)

    # Perform Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    # Convert expected frequencies to DataFrame for display and residual calculation
    expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
    print("\n📈 Expected Frequencies (Interpretation × Syntax) (Rounded to 2 decimal places):")
    print(expected_df.round(2))

    # Calculate Standardized Pearson Residuals
    residuals = (contingency_table - expected_df) / np.sqrt(expected_df)
    print("\n🔥 Standardized Pearson Residuals (Interpretation × Syntax) (Rounded to 2 decimal places):")
    print(residuals.round(2))

    # Cramér's V calculation (Effect Size)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    cramers_v = np.sqrt(phi2 / min(k - 1, r - 1)) if min(k - 1, r - 1) > 0 else 0

    # Output Results
    print(f"\n📊 Chi-square Test Results (Syntax × Interpretation):")
    print("------------------------------------------")
    print(f"Chi² Statistic      : {chi2:.4f}")
    print(f"Degrees of Freedom  : {dof}")
    print(f"P-value             : {p:.4e}")
    print(f"Cramér's V          : {cramers_v:.4f}")


    # Define custom color map based on residual sign, scaled to max residual of 3 for color intensity
    def color_map(val):
        """Maps a residual value to a color from Blues (positive) or Reds (negative)."""
        if val > 0:
            return plt.cm.Blues(min(val / 3, 1))
        else:
            return plt.cm.Reds(min(-val / 3, 1))

    # Plotting the grouped bar chart for Syntax vs. Interpretation
    fig, ax = plt.subplots(figsize=(12, 8)) # Adjust figure size for better readability
    bar_width = 0.25 # Width of each individual bar
    
    # Get interpretation categories (rows) and syntax categories (columns)
    interpretation_categories = contingency_table.index.tolist() # e.g., Basic, Emotion, Nonliteral
    syntax_categories = contingency_table.columns.tolist() # e.g., Attributive, Predicative

    # Create an array of indices for x-axis positions of syntax groups
    indices = np.arange(len(syntax_categories))

    # Plot bars for each interpretation category
    num_interpretations = len(interpretation_categories)
    for i, cat in enumerate(interpretation_categories):
        # Calculate positions for bars within each syntax group
        # This offset correctly positions the bars for each interpretation category
        offset = (i - num_interpretations / 2 + 0.5) * bar_width
        positions = indices + offset
        
        # Get residuals for current interpretation category across all syntaxes
        vals = residuals.loc[cat].values
        colors = [color_map(v) for v in vals] # Apply custom color map based on residuals

        ax.bar(positions, vals, width=bar_width, color=colors, edgecolor='black', label=cat)

        # Annotate bars with actual observed counts (from contingency_table)
        counts = contingency_table.loc[cat].values
        for x_bar, y_bar, count_val in zip(positions, vals, counts):
            # Adjust text vertical position based on residual sign
            text_y_offset = 0.05 * np.sign(y_bar) if y_bar != 0 else 0.1 # Small offset, avoid zero residual text overlap
            ax.text(x_bar, y_bar + text_y_offset, str(int(count_val)),
                    ha='center', va='bottom' if y_bar >= 0 else 'top', fontsize=9, color='black')

    # Formatting the plot
    # Set x-ticks to center of each syntax group
    ax.set_xticks(indices)
    ax.set_xticklabels(syntax_categories) # Labels for syntactic positions
    
    ax.axhline(0, color='black', linewidth=0.8) # Horizontal line at zero residual
    ax.set_ylabel('Standardized Pearson Residuals', fontsize=12)
    ax.set_title('Association Plot: Interpretation × Syntax (Standardized Residuals)', fontsize=16)
    ax.legend(title='Meaning', bbox_to_anchor=(1.05, 1), loc='upper left') # Legend title matches image

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show() # Display the plot

    # --- Optional: Save Tables and Plot ---
    # contingency_table.to_csv("observed_frequencies_syntax_interpretation.csv")
    # expected_df.to_csv("expected_frequencies_syntax_interpretation.csv")
    # residuals.to_csv("standardized_residuals_syntax_interpretation.csv")
    # fig.savefig("association_plot_syntax_interpretation.png", dpi=300, bbox_inches='tight')


# --- 3. Execute Analyses for Specific Pairs ---

# Analysis for Genre vs. Interpretation (using the generic horizontal bar chart helper function)
analyze_and_plot_association(df_clean, 'genre', 'interpretation', "Genre × Interpretation")

# Analysis for Decade vs. Interpretation (using the generic horizontal bar chart helper function)
analyze_and_plot_association(df_clean, 'decade', 'interpretation', "Decade × Interpretation")

# Analysis for Syntax vs. Interpretation (using the specific grouped bar chart helper function)
plot_syntax_vs_interpretation_grouped_bars(df_clean)
