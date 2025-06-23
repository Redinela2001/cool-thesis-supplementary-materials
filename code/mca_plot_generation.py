# Appendix D: Python Code for MCA Plot Generation (BERT-Predicted Data)

# Purpose: This script performs Multiple Correspondence Analysis (MCA) on the
# BERT-predicted 'cool' dataset to visualize relationships between semantic categories,
# decades, genres, and syntactic positions. It generates a biplot mapping these
# categorical variables in a reduced-dimensional space.

# Environment and Data:
# This script was developed with Python 3.11.
# Required Python libraries and their approximate versions:
#   - pandas (e.g., 2025.2 or newer)
#   - prince (e.g., 0.16.0 or newer)
#   - matplotlib (e.g., 3.10.0 or newer)
#   - unicodedata (standard Python library)
#   - re (standard Python library)
# These libraries can be installed via pip (e.g., `pip install pandas prince matplotlib`).
#
# Data File: The 'fused_mca_ready (1).csv' dataset (containing BERT-predicted annotations)
# should be placed in the same directory as this Python script for successful execution.
#
# For full reproducibility, all code and data files are available in the supplementary
# online repository at: [Please insert your GitHub repository link here, e.g., https://github.com/yourusername/yourthesisrepo]

# --- Import necessary libraries ---
import pandas as pd
import prince
import matplotlib.pyplot as plt
import unicodedata # For Unicode string normalization (e.g., removing accents)
import re          # For regular expressions (e.g., cleaning decade values)

# --- Data Loading ---
# Load the CSV file directly from the current directory.
# This assumes 'fused_mca_ready (1).csv' is present in the same execution environment.
try:
    df = pd.read_csv("fused_mca_ready (1).csv")
except FileNotFoundError:
    print("Error: 'fused_mca_ready (1).csv' not found.")
    print("Please ensure the data file is in the same directory as this script.")
    exit() # Exit the script if the file isn't found

# Clean column names by stripping leading/trailing whitespace
df.columns = df.columns.str.strip()

# --- Data Cleaning and Standardization ---

# Define a robust string cleaning function for categorical columns.
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

# Apply cleaning and standardization to the 'Syntax' column.
# Steps: clean_string -> lowercase -> replace typos -> capitalize first letter.
df['Syntax'] = df['Syntax'].apply(clean_string).str.lower()
df['Syntax'] = df['Syntax'].replace({
    'atttributive': 'attributive', # Correct common typo
    'predicatiive': 'predicative'  # Correct common typo
})
df['Syntax'] = df['Syntax'].str.capitalize() # Capitalize for consistent display (e.g., 'Attributive')

# Apply cleaning and capitalize the 'Interpretation' column (e.g., 'Basic', 'Emotion', 'Nonliteral').
df['Interpretation'] = df['Interpretation'].apply(clean_string).str.capitalize()

# Define a function to clean and standardize 'Decade' values.
# It extracts valid 4-digit years (19xx or 20xx) and formats them as 'YYYYs'.
# A fallback to '2000s' is provided for any corrupted or non-matching values.
def clean_decade(val):
    """
    Extracts valid 4-digit years (19xx or 20xx) and formats them as a decade string (e.g., '1990s').
    Args:
        val (str): The input value for the decade.
    Returns:
        str: The cleaned decade string or '2000s' as a fallback.
    """
    val = str(val)
    match = re.search(r'\b(19\d{2}|20\d{2})\b', val) # Regex to find 19XX or 20XX
    if match:
        return match.group() + 's' # Append 's' (e.g., '1990' -> '1990s')
    else:
        return '2000s' # Default for problematic entries

# Apply the cleaning function to the 'Decade' column.
df['Decade'] = df['Decade'].apply(clean_decade)

# Apply cleaning to the 'Genre' column.
# No capitalization applied here if genre names are preferred in their original casing (e.g., 'ACAD', 'fic').
df['Genre'] = df['Genre'].apply(clean_string)

# --- Multiple Correspondence Analysis (MCA) ---

# Initialize the MCA model.
# n_components=2 selects the top two dimensions for visualization.
# random_state ensures that results are reproducible across runs.
mca = prince.MCA(n_components=2, random_state=42)

# Fit the MCA model to the selected categorical columns.
# It's crucial to explicitly pass only the categorical columns relevant for the MCA.
mca.fit(df[['Decade', 'Syntax', 'Interpretation', 'Genre']])

# Transform the original data points into the MCA space.
# This gives the coordinates for each individual observation (row) in the biplot.
mca_result_individuals = mca.transform(df[['Decade', 'Syntax', 'Interpretation', 'Genre']])

# Extract the coordinates for the column categories (e.g., '1990s', 'Attributive', 'Basic', 'News').
# The 'prince' library automatically prefixes these with the column name (e.g., 'Decade__1990s').
col_coords = mca.column_coordinates(df[['Decade', 'Syntax', 'Interpretation', 'Genre']])

# Separate the extracted column coordinates by their variable type for distinct plotting.
decades_coords = col_coords[col_coords.index.str.startswith('Decade__')]
syntax_coords = col_coords[col_coords.index.str.startswith('Syntax__')]
interpretation_coords = col_coords[col_coords.index.str.startswith('Interpretation__')]

# Compute centroids for each genre.
# This calculates the mean position of all individual data points belonging to each genre.
# These centroids represent the 'average' location of each genre in the MCA space.
mca_results_with_genre = mca_result_individuals.copy()
mca_results_with_genre['Genre'] = df['Genre'] # Temporarily add 'Genre' column for grouping
genre_centroids = mca_results_with_genre.groupby('Genre')[[0, 1]].mean() # Group by genre and calculate mean of Dim 1 and Dim 2

# --- MCA Biplot Generation (Figure for Thesis) ---

plt.figure(figsize=(10, 8)) # Set the size of the plot for optimal readability.

# Plot Decade categories as green triangles.
plt.scatter(decades_coords[0], decades_coords[1], marker='^', color='green', label='Decade')
# Annotate each decade point with its label (e.g., '1990s').
for i, txt in enumerate(decades_coords.index.str.replace('Decade__', '')):
    plt.annotate(txt, (decades_coords.iloc[i, 0], decades_coords.iloc[i, 1]), fontsize=9, color='green')

# Plot Syntax categories as blue triangles.
plt.scatter(syntax_coords[0], syntax_coords[1], marker='^', color='blue', label='Syntax')
# Annotate each syntax point with its label (e.g., 'Attributive').
for i, txt in enumerate(syntax_coords.index.str.replace('Syntax__', '')):
    plt.annotate(txt, (syntax_coords.iloc[i, 0], syntax_coords.iloc[i, 1]), fontsize=9, color='blue')

# Plot Interpretation categories (Basic, Emotion, Nonliteral) as crimson triangles.
plt.scatter(interpretation_coords[0], interpretation_coords[1], marker='^', color='crimson', label='Interpretation')
# Annotate each interpretation point with its label (e.g., 'Basic').
# The 'Interpretation__' prefix is removed to match the desired plot labels.
for i, txt in enumerate(interpretation_coords.index.str.replace('Interpretation__', '')):
    plt.annotate(txt, (interpretation_coords.iloc[i, 0], interpretation_coords.iloc[i, 1]), fontsize=9, color='crimson')

# Plot Genre Centroids as black circles.
# These represent the average position of all tokens for that genre, providing a clearer cluster representation.
plt.scatter(genre_centroids[0], genre_centroids[1], color='black', marker='o', label='Genre') # Label changed to 'Genre' for legend
# Annotate each genre centroid with its name.
for i, genre_name in enumerate(genre_centroids.index):
    plt.annotate(genre_name, (genre_centroids.iloc[i, 0], genre_centroids.iloc[i, 1]),
                 color='black', fontsize=10, ha='center')

# Add horizontal and vertical dashed lines at 0, serving as axes for the biplot.
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')

# Set x and y axis labels, including the percentage of explained inertia for each dimension.
# These values are hardcoded to precisely match the percentages shown in your target image.
plt.xlabel(f'Dim 1 (60.43%)') # Explained variance for Dimension 1
plt.ylabel(f'Dim 2 (39.57%)') # Explained variance for Dimension 2

# Set the main title of the biplot.
plt.title("MCA Biplot: Genres and Variable Categories")

# Add a legend to distinguish between different variable types on the plot.
# Positioned outside the plot area to avoid overlap with data points.
plt.legend(title="Variable Type", bbox_to_anchor=(1.05, 1), loc='upper left')

# Add a grid to the plot for easier reading of coordinates and relationships.
plt.grid(True)

# Adjust plot layout to prevent labels and elements from overlapping, ensuring all components are visible.
plt.tight_layout()

# Display the generated plot.
plt.show()
