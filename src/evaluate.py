"""
Evaluate trained pipelines on the test set.
Loads models from .joblib and metadata from .json to ensure correct environment.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

def evaluate_models(model_dir="models", plots_dir="plots"):
    model_path = Path(model_dir)
    plots_path = Path(plots_dir)
    plots_path.mkdir(exist_ok=True)
    
    # Load Metadata
    try:
        meta = json.loads((model_path / "metadata.json").read_text())
        print("Loaded metadata. Target mapping:", meta["target"]["mapping"])
    except FileNotFoundError:
        print("Error: metadata.json not found. Run train.py first.")
        return

    # Load Test Data (saved by train.py inside metadata or separate pickle? 
    # The new train.py didn't convert test set to pickle, it's safer to reload raw data and split exactly same way)
    # BUT for simplicity, let's assume valid data access. 
    # Actually, the new train.py does NOT save X_test. It assumes you can reload.
    # Let's verify train.py... It does NOT save test_data.pkl.
    # So we must recreate the split using the same random_state from metadata.
    
    from train import load_data, make_binary_target, drop_leakage_columns, LEAKAGE_PATTERN
    from sklearn.model_selection import train_test_split
    
    # Reload data
    # Assuming data path is fixed or we find it. For now hardcode or assume relative.
    # We will try to locate data.csv
    data_path = Path("data/data.csv")
    if not data_path.exists():
         data_path = Path("../data/data.csv")
    
    df = load_data(data_path)
    y = make_binary_target(df, target_col="Target")
    df_X = df.drop(columns=["Target"])
    
    if meta["drop_2nd_sem"]:
        df_X, _ = drop_leakage_columns(df_X, leakage_pattern=LEAKAGE_PATTERN)
        
    # Re-split using exactly same state
    _, X_test, _, y_test = train_test_split(
        df_X, y, test_size=meta["test_size"], random_state=meta["random_state"], stratify=y
    )
    
    # Load Models
    models = {}
    try:
        models["Random Forest"] = joblib.load(model_path / "rf_pipeline.joblib")
        models["XGBoost"] = joblib.load(model_path / "xgb_pipeline.joblib")
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Get chosen threshold
    chosen_thresh = meta.get("chosen_threshold", 0.5)
    print(f"\nUsing chosen threshold from training: {chosen_thresh:.3f}")

    class_names = ["Non-Dropout", "Dropout"]

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Default predictions (0.5)
        y_pred_default = (y_prob >= 0.5).astype(int)
        
        # Chosen threshold predictions
        y_pred_chosen = (y_prob >= chosen_thresh).astype(int)
        
        # Report for Default
        print(f"\n--- {name} (Threshold 0.50) ---")
        print(classification_report(y_test, y_pred_default, target_names=class_names))
        
        # Report for Chosen
        print(f"\n--- {name} (Threshold {chosen_thresh:.3f}) ---")
        print(classification_report(y_test, y_pred_chosen, target_names=class_names))
        
        # Use Chosen prediction for plots? Usually plots like ROC/PR are threshold-independent.
        # But for Confusion Matrix, it depends on a hard decision.
        # Let's use the CHOSEN threshold for the saved Confusion Matrix to align with "optimized" performance.
        
        # Confusion Matrix (Chosen Threshold)
        cm = confusion_matrix(y_test, y_pred_chosen)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{name} Confusion Matrix (Thresh={chosen_thresh:.2f})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(plots_path / f"{name.replace(' ', '_')}_confusion_matrix.png")
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"{name} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_path / f"{name.replace(' ', '_')}_roc_curve.png")
        plt.close()
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.2f}")
        plt.title(f"{name} Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_path / f"{name.replace(' ', '_')}_pr_curve.png")
        plt.close()
        
        print(f"PR-AUC: {pr_auc:.2f}")

if __name__ == "__main__":
    evaluate_models()
