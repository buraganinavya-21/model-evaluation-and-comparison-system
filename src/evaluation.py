"""
Model Evaluation Module
=======================
Trains selected classification models and calculates evaluation metrics:
Accuracy, Precision, Recall, F1-Score, ROC-AUC score, Confusion Matrix,
and full Classification Report.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_single_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list
) -> dict:
    """
    Trains a model and computes comprehensive classification metrics.
    """
    num_classes = len(class_names)
    average_mode = 'binary' if num_classes == 2 else 'weighted'
    
    # Fit Model
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Core Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average=average_mode, zero_division=0))
    recall = float(recall_score(y_test, y_pred, average=average_mode, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average=average_mode, zero_division=0))
    
    # ROC-AUC Calculation where applicable
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if num_classes == 2:
                roc_auc = float(roc_auc_score(y_test, y_prob[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted'))
        except Exception:
            roc_auc = None
            
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification Report
    clf_report_dict = classification_report(
        y_test, y_pred, target_names=[str(c) for c in class_names], output_dict=True, zero_division=0
    )
    clf_report_df = pd.DataFrame(clf_report_dict).transpose()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report_df': clf_report_df,
        'y_pred': y_pred
    }


def train_and_evaluate_all(
    selected_names: list,
    available_models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list
) -> tuple:
    """
    Runs evaluation for all user-selected models.
    
    Returns:
        (results_dict, summary_df)
    """
    results_dict = {}
    summary_rows = []
    
    for name in selected_names:
        if name not in available_models:
            continue
            
        model = available_models[name]
        try:
            res = evaluate_single_model(model, X_train, y_train, X_test, y_test, class_names)
            results_dict[name] = res
            
            row = {
                'Model': name,
                'Accuracy': res['accuracy'],
                'Precision': res['precision'],
                'Recall': res['recall'],
                'F1-Score': res['f1_score'],
                'ROC-AUC': res['roc_auc'] if res['roc_auc'] is not None else np.nan
            }
            summary_rows.append(row)
        except Exception as e:
            results_dict[name] = {'error': str(e)}
            
    summary_df = pd.DataFrame(summary_rows)
    return results_dict, summary_df
