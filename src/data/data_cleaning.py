import pandas as pd
import os
from utils.project_functions import load_data

def preprocess_data(data):
    """
    Handles missing values, ensures consistency in data types. Return preprocessed data as dataframe.
    """
    try:
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        print("Data preprocessing completed successfully")
        return data
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return None

if __name__ == "__main__":
    file_path = "data/raw/bitcoin_data.csv"
    print(f"Loading data from {file_path}...")
    data = load_data(file_path)
    if data is not None:
        print("Data loaded. Starting preprocessing...")
        preprocessed_data = preprocess_data(data)
        if preprocessed_data is not None:
            output_path = "data/processed/preprocessed_data.csv"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            preprocessed_data.to_csv(output_path, index=False)
            print(f"Preprocessed data saved to {output_path}")
        else:
            print("Preprocessing failed.")
    else:
        print("Data loading failed.")