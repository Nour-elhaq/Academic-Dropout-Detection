import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from data_loader import load_data

def preprocess_data(target_column='Target', test_size=0.2, random_state=42):
    df = load_data()
    
    # --- 1. REMOVE LEAKAGE (2nd Semester Features) ---
    # We drop any column related to the 2nd semester to prevent future information leakage.
    leakage_cols = [col for col in df.columns if '2nd sem' in col]
    print(f"Dropping {len(leakage_cols)} leakage columns (2nd Semester)...")
    X = df.drop(target_column, axis=1).drop(columns=leakage_cols)
    y = df[target_column]
    
    # --- 2. REFRAME PROBLEM (Binary Classification) ---
    # Target: 1 if "Dropout" (At Risk), 0 otherwise (Enrolled/Graduate)
    print("Reframing problem to Binary Classification: Dropout (1) vs Non-Dropout (0)")
    y_binary = y.apply(lambda x: 1 if x == 'Dropout' else 0)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle Class Imbalance (SMOTE) on Training Data
    # For binary classification, this balances the 1s and 0s
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test, None # No label encoder needed for binary integers

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, le = preprocess_data()
    print("Data Preprocessing Complete.")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # print(f"Classes: {le.classes_}") # le is None in binary mode
