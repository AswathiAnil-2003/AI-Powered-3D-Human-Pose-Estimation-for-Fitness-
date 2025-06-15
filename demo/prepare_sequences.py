import os
import numpy as np
import pandas as pd

INPUT_FOLDER = "data/processed/normalized"
OUTPUT_FOLDER = "data/sequences"
SEQUENCE_LENGTH = 30

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def create_sequences(file_path, sequence_length=30):
    df = pd.read_csv(file_path)
    data = df.drop(columns=['frame'], errors='ignore').values

    sequences = []
    for i in range(len(data) - sequence_length + 1):
        window = data[i:i + sequence_length]
        sequences.append(window)

    return np.array(sequences)

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".csv"):
        filepath = os.path.join(INPUT_FOLDER, filename)

        # ✅ Skip empty files
        if os.path.getsize(filepath) == 0:
            print(f"⚠ Skipping empty file: {filename}")
            continue

        base_name = filename.replace("_normalized.csv", "")
        npy_filename = f"{base_name}_seq{SEQUENCE_LENGTH}.npy"
        npy_path = os.path.join(OUTPUT_FOLDER, npy_filename)

        # ✅ Skip if .npy already exists
        if os.path.exists(npy_path):
            print(f"Skipping already converted: {npy_filename}")
            continue

        try:
            sequences = create_sequences(filepath, SEQUENCE_LENGTH)
            np.save(npy_path, sequences)
            print(f"Saved: {npy_filename} | Shape: {sequences.shape}")
        except pd.errors.EmptyDataError:
            print(f"⚠ Skipping invalid/empty CSV (pandas error): {filename}")
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")
