import pandas as pd
import os

def load_data(filepath='data/data.csv'):
    """
    Loads the student dropout dataset.
    """
    if not os.path.exists(filepath):
        # Fallback to absolute path or trying different relative paths if run from src
        if os.path.exists('../data/data.csv'):
           filepath = '../data/data.csv'
        else:
           raise FileNotFoundError(f"File not found at {filepath}")

    df = pd.read_csv(filepath, sep=';')
    
    # Rename columns with typos or extra spaces
    df.rename(columns={
        'Nacionality': 'Nationality',
        'Daytime/evening attendance\t': 'Daytime/evening attendance',
        'Displaced': 'Displaced', # checking if rename is needed, seems ok
    }, inplace=True)
    
    return df

if __name__ == "__main__":
    try:
        df = load_data()
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
