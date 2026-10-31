"""
Data Preprocessing Pipeline Module
====================================
Handles dataset loading, dataset statistics inspection, target column validation,
and data leakage-free preprocessing pipelines using Scikit-Learn.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Computes key summary statistics for the dataset.
    
    Returns:
        dict containing rows, columns, missing values count, duplicate rows count,
        and column data types breakdown.
    """
    num_rows, num_cols = df.shape
    missing_total = int(df.isnull().sum().sum())
    missing_by_col = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    
    # Feature type breakdown
    num_cols_list = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_list = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    dtype_summary = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Missing Values': df.isnull().sum().values,
        'Missing (%)': (df.isnull().sum().values / num_rows * 100).round(2),
        'Unique Values': df.nunique().values
    })
    
    return {
        'num_rows': num_rows,
        'num_cols': num_cols,
        'missing_total': missing_total,
        'missing_by_col': missing_by_col,
        'duplicate_rows': duplicate_rows,
        'num_cols_list': num_cols_list,
        'cat_cols_list': cat_cols_list,
        'dtype_summary': dtype_summary
    }


def validate_target(df: pd.DataFrame, target_col: str) -> dict:
    """
    Validates if the chosen target column is suitable for classification.
    
    Returns:
        dict with status ('valid', 'warning', 'error'), message, and class distribution.
    """
    if target_col not in df.columns:
        return {
            'status': 'error',
            'message': f"Column '{target_col}' not found in dataset.",
            'is_valid': False
        }
    
    target_series = df[target_col].dropna()
    unique_vals = target_series.nunique()
    
    if unique_vals < 2:
        return {
            'status': 'error',
            'message': f"Target '{target_col}' has only {unique_vals} unique class. Classification requires at least 2 distinct classes.",
            'is_valid': False
        }
    
    # Check if target looks continuous (e.g., float with many unique values)
    is_float = pd.api.types.is_float_dtype(target_series)
    if is_float and unique_vals > 20:
        return {
            'status': 'warning',
            'message': f"Target '{target_col}' has {unique_vals} unique numeric floating-point values. It appears to be continuous (Regression target). Classification models might perform poorly or produce too many classes.",
            'is_valid': True,
            'unique_count': unique_vals,
            'class_distribution': target_series.value_counts()
        }
    
    return {
        'status': 'valid',
        'message': f"Target '{target_col}' is suitable for classification with {unique_vals} classes.",
        'is_valid': True,
        'unique_count': unique_vals,
        'class_distribution': target_series.value_counts()
    }


def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    remove_duplicates: bool = False,
    random_state: int = 42
) -> dict:
    """
    Preprocesses dataset without data leakage.
    1. Removes duplicates if requested.
    2. Separates input features X and target y.
    3. Encodes target y with LabelEncoder.
    4. Splits into train and test sets.
    5. Fits ColumnTransformer (imputing + scaling numerical, imputing + one-hot encoding categorical) strictly on X_train.
    6. Transforms X_train and X_test.
    
    Returns:
        dict with preprocessed arrays, feature names, encoders, and class mappings.
    """
    df_clean = df.copy()
    
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()
    
    # Drop rows where target is missing
    df_clean = df_clean.dropna(subset=[target_col])
    
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # Target Encoding
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))
    class_names = target_encoder.classes_.tolist()
    
    # Identify feature types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Train / Test Split FIRST to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
    )
    
    # Define ColumnTransformer Transformers
    transformers = []
    
    if num_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, num_cols))
        
    if cat_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, cat_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Fit preprocessor strictly on X_train, then transform X_train and X_test
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Extract feature names after encoding
    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        try:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
            feature_names.extend(encoded_cat_names)
        except Exception:
            feature_names.extend([f"cat_{i}" for i in range(X_train_proc.shape[1] - len(num_cols))])
            
    return {
        'X_train_proc': X_train_proc,
        'X_test_proc': X_test_proc,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'target_encoder': target_encoder,
        'class_names': class_names,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'train_samples': X_train_proc.shape[0],
        'test_samples': X_test_proc.shape[0],
        'num_features_processed': X_train_proc.shape[1]
    }
