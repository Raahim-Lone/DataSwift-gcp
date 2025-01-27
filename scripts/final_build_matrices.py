import pandas as pd
import numpy as np
import os

# Define file paths
query_latency_test_path = '/Users/raahimlone/rahhh/Data_Gathering/query_latency_test.csv'
masked_path = '/Users/raahimlone/rahhh/Data_Gathering/masked.csv'
output_dir = '/Users/raahimlone/New_Data'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

try:
    # ----------------------------
    # Part 1: Create W_full Matrix
    # ----------------------------

    # Load query_latency_test.csv
    df_full = pd.read_csv(query_latency_test_path)
    print("Loaded query_latency_test.csv successfully.")

    # Check columns
    print("Columns in query_latency_test.csv:", df_full.columns.tolist())

    # Handle missing values
    df_full = df_full.dropna(subset=['query', 'hint', 'latency_seconds'])
    print("Dropped rows with missing values in W_full.")

    # Ensure latency is numeric
    df_full['latency_seconds'] = pd.to_numeric(df_full['latency_seconds'], errors='coerce')
    df_full = df_full.dropna(subset=['latency_seconds'])
    print("Converted latency to numeric and dropped NaNs.")

    # Extract unique queries and hints
    queries_full = df_full['query'].unique()
    hints_full = df_full['hint'].unique()
    print(f"Number of unique queries in W_full: {len(queries_full)}")
    print(f"Number of unique hints in W_full: {len(hints_full)}")

    # Create mappings
    query_to_idx_full = {query: idx for idx, query in enumerate(queries_full)}
    hint_to_idx_full = {hint: idx for idx, hint in enumerate(hints_full)}

    # Initialize W_full matrix with zeros
    W_full = np.zeros((len(queries_full), len(hints_full)))
    print("Initialized W_full matrix with zeros.")

    # Populate W_full matrix
    for _, row in df_full.iterrows():
        q_idx = query_to_idx_full[row['query']]
        h_idx = hint_to_idx_full[row['hint']]
        W_full[q_idx, h_idx] = row['latency_seconds']
    print("Populated W_full matrix with latency values.")

    # Save W_full matrix
    W_full_path = os.path.join(output_dir, 'W_full.npy')
    np.save(W_full_path, W_full)
    print(f"W_full matrix saved to {W_full_path}")

    # ----------------------------
    # Part 2: Create W and M Matrices
    # ----------------------------

    # Load masked.csv
    df_masked = pd.read_csv(masked_path)
    print("Loaded masked.csv successfully.")

    # Check columns
    print("Columns in masked.csv:", df_masked.columns.tolist())

    # Handle missing values
    df_masked = df_masked.dropna(subset=['query', 'hint', 'latency_seconds'])
    print("Dropped rows with missing values in masked data.")

    # Ensure latency is numeric
    df_masked['latency_seconds'] = pd.to_numeric(df_masked['latency_seconds'], errors='coerce')
    df_masked = df_masked.dropna(subset=['latency_seconds'])
    print("Converted latency to numeric and dropped NaNs in masked data.")

    # Extract unique queries and hints
    queries_masked = df_masked['query'].unique()
    hints_masked = df_masked['hint'].unique()
    print(f"Number of unique queries in masked data: {len(queries_masked)}")
    print(f"Number of unique hints in masked data: {len(hints_masked)}")

    # Create mappings
    query_to_idx_masked = {query: idx for idx, query in enumerate(queries_masked)}
    hint_to_idx_masked = {hint: idx for idx, hint in enumerate(hints_masked)}

    # Initialize W and M matrices
    W = np.zeros((len(queries_masked), len(hints_masked)))
    M = np.zeros((len(queries_masked), len(hints_masked)), dtype=int)
    print("Initialized W and M matrices with zeros.")

    # Populate W and M matrices
    for _, row in df_masked.iterrows():
        q_idx = query_to_idx_masked[row['query']]
        h_idx = hint_to_idx_masked[row['hint']]
        W[q_idx, h_idx] = row['latency_seconds']
        M[q_idx, h_idx] = 1  # Mark as observed
    print("Populated W and M matrices.")

    # Save W and M matrices
    W_path = os.path.join(output_dir, 'W.npy')
    M_path = os.path.join(output_dir, 'M.npy')
    np.save(W_path, W)
    np.save(M_path, M)
    print(f"W matrix saved to {W_path}")
    print(f"M matrix saved to {M_path}")

    # Optional: Save all matrices into a single .npz file
    npz_path = os.path.join(output_dir, 'matrices.npz')
    np.savez(npz_path, W_full=W_full, W=W, M=M)
    print(f"All matrices saved to {npz_path}")

except FileNotFoundError as fnf_error:
    print(f"File not found: {fnf_error.filename}")
except KeyError as key_error:
    print(f"Missing expected column: {key_error}")
except ValueError as val_error:
    print(f"Value error: {val_error}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
