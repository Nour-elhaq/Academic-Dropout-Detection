import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid Qt errors
import seaborn as sns
import matplotlib.pyplot as plt
import os
from data_loader import load_data

def perform_eda():
    df = load_data()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    print("Generating EDA plots...")

    # 1. Target Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Target', data=df)
    plt.title('Target Class Distribution')
    plt.savefig('plots/target_distribution.png')
    plt.close()

    # 2. Correlation Matrix (Numerical features)
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

    # 3. Age Distribution by Target
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Age at enrollment', hue='Target', fill=True, common_norm=False)
    plt.title('Age Distribution by Target Status')
    plt.savefig('plots/age_distribution_by_target.png')
    plt.close()
    
    # 4. Gender vs Target
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Gender', hue='Target', data=df)
    plt.title('Gender vs Target (0=Female, 1=Male)')
    plt.savefig('plots/gender_vs_target.png')
    plt.close()

    print("EDA plots saved in 'plots/' directory.")

if __name__ == "__main__":
    perform_eda()
