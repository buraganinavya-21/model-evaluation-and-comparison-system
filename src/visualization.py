"""
Visualization Module
====================
Provides clean, readable, professional Matplotlib and Seaborn plots for:
- Target Class Distribution
- Comparative Model Metrics (Grouped Bar Chart & Accuracy Bar Chart)
- Confusion Matrix Heatmaps
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set clean light Seaborn theme for academic presentation
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor': '#FFFFFF',
    'text.color': '#0F172A',
    'axes.labelcolor': '#0F172A',
    'xtick.color': '#0F172A',
    'ytick.color': '#0F172A'
})


def plot_target_distribution(class_distribution: pd.Series, target_col: str):
    """
    Generates a bar chart showing target class distribution.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    categories = [str(c) for c in class_distribution.index]
    counts = class_distribution.values
    
    colors = sns.color_palette("Blues_d", len(categories))
    bars = ax.bar(categories, counts, color=colors, edgecolor="black", linewidth=0.8)
    
    ax.set_title(f"Target Class Distribution: '{target_col}'", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Class Label", fontsize=10, labelpad=8)
    ax.set_ylabel("Sample Count", fontsize=10, labelpad=8)
    
    # Annotate bar counts
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    return fig


def plot_metrics_comparison(summary_df: pd.DataFrame):
    """
    Generates a grouped bar chart comparing Accuracy, Precision, Recall, and F1-Score.
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    df_melted = summary_df.melt(id_vars=['Model'], value_vars=metrics, var_name='Metric', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(
        data=df_melted,
        x='Model',
        y='Score',
        hue='Metric',
        palette='magma',
        ax=ax,
        edgecolor="black",
        linewidth=0.5
    )
    
    ax.set_title("Classification Metrics Comparison Across Algorithms", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Algorithm", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylim(0, 1.12)
    ax.legend(title="Metric", loc='upper right', frameon=True)
    plt.xticks(rotation=15, ha='right', fontsize=9.5)
    
    # Value annotations on bars
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(p.get_x() + p.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7.5, rotation=90)
                        
    plt.tight_layout()
    return fig


def plot_accuracy_ranking(summary_df: pd.DataFrame):
    """
    Generates a horizontal bar chart displaying ranked Model Accuracies.
    """
    df_sorted = summary_df.sort_values(by='Accuracy', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = sns.color_palette("crest", len(df_sorted))
    bars = ax.barh(df_sorted['Model'], df_sorted['Accuracy'], color=colors, edgecolor="black", linewidth=0.8)
    
    ax.set_title("Model Accuracy Ranking", fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Accuracy Score", fontsize=10, labelpad=8)
    ax.set_xlim(0, 1.15)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.4f} ({width*100:.1f}%)',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: list, model_name: str):
    """
    Plots an annotated confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = [str(c) for c in class_names]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=1,
        linecolor='black'
    )
    
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel("Actual (True) Label", fontsize=10, fontweight='bold', labelpad=8)
    
    plt.tight_layout()
    return fig
