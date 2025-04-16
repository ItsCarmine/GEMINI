import pandas as pd
import os
import numpy as np # Import numpy for np.nan comparison if needed

# --- Configuration ---
INPUT_FOLDER = "original_datasets"
OUTPUT_FOLDER = "processed_datasets" # Create this folder if it doesn't exist
INPUT_CSV_NAME = "gz2_hart16.csv"
OUTPUT_CSV_NAME = "gz2_simplified_labels.csv"

# Define the columns we need from the input CSV
ID_COLUMN = "dr7objid"
SMOOTH_COL = "t01_smooth_or_features_a01_smooth_debiased"
FEATURED_COL = "t01_smooth_or_features_a02_features_or_disk_debiased"
ARTIFACT_COL = "t01_smooth_or_features_a03_star_or_artifact_debiased"

# Threshold for classification confidence
# A galaxy is assigned a class if its debiased vote fraction exceeds this
THRESHOLD = 0.8

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Construct file paths ---
input_csv_path = os.path.join(INPUT_FOLDER, INPUT_CSV_NAME)
output_csv_path = os.path.join(OUTPUT_FOLDER, OUTPUT_CSV_NAME)

# --- Load the dataset ---
print(f"Reading input CSV: {input_csv_path}...")
try:
    df = pd.read_csv(input_csv_path)
    print(f"Successfully loaded {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: Input file not found at {input_csv_path}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Define the simplification function ---
def simplify_classification(row):
    """
    Assigns a simplified class based on debiased vote fractions.
    Priority: Smooth > Featured > Artifact.
    """
    # Check for NaN values, return 'Uncertain' if critical data is missing
    if pd.isna(row[SMOOTH_COL]) or pd.isna(row[FEATURED_COL]) or pd.isna(row[ARTIFACT_COL]):
        return "Uncertain"

    if row[SMOOTH_COL] >= THRESHOLD:
        return "Smooth"
    elif row[FEATURED_COL] >= THRESHOLD:
        # You could further subdivide 'Featured' later using other columns
        # (e.g., edge-on, spiral) if needed for a more complex project.
        return "Featured"
    elif row[ARTIFACT_COL] >= THRESHOLD:
        return "Artifact"
    else:
        # None of the primary categories reached the threshold
        return "Uncertain"

# --- Apply the function to create the new classification column ---
print("Applying classification simplification...")
df["simplified_class"] = df.apply(simplify_classification, axis=1)

# --- Select relevant columns for the output ---
output_df = df[[ID_COLUMN, "simplified_class"]].copy()

# --- Save the simplified dataset ---
print(f"Saving simplified labels to: {output_csv_path}...")
output_df.to_csv(output_csv_path, index=False)

# --- Print class distribution ---
print("\n--- Simplified Class Distribution ---")
print(output_df["simplified_class"].value_counts())
print("------------------------------------")

print(f"\nProcessing complete. Simplified labels saved.")

# --- Note on Image Matching ---
# The 'dr7objid' column in the output CSV should correspond to the
# filenames of the images downloaded from Zenodo (e.g., 587722984761671716.jpg).
# You will use 'gz2_simplified_labels.csv' to link images to their
# 'Smooth', 'Featured', 'Artifact', or 'Uncertain' label during model training.
# The 'gz2_filename_mappings.csv' might not be needed if the image filenames
# directly match the dr7objid.