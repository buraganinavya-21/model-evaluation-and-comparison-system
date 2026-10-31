"""
Machine Learning Models Module
==============================
Defines classification algorithms and provides educational documentation
for B.Tech capstone project presentation and Viva examination.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def get_available_models(random_state: int = 42) -> dict:
    """
    Instantiates and returns dictionary of supported classification models.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "Support Vector Machine (SVM)": SVC(
            probability=True,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        )
    }
    return models


def get_model_insights() -> dict:
    """
    Returns educational summaries and algorithm insights for model comparison reference.
    """
    info = {
        "Logistic Regression": {
            "name": "Logistic Regression",
            "type": "Linear Model",
            "principle": "Uses the sigmoid function to map linear combinations of input features into probabilities between 0 and 1, applying a threshold (usually 0.5) for classification.",
            "hyperparameters": "C (inverse regularization strength), penalty ('l1', 'l2', 'elasticnet'), solver ('lbfgs', 'liblinear').",
            "advantages": "Simple, fast to train, highly interpretable, outputs clear class probabilities, works well on linearly separable data.",
            "limitations": "Assumes linear relationship between features and log-odds of target; struggles with complex non-linear patterns."
        },
        "K-Nearest Neighbors (KNN)": {
            "name": "K-Nearest Neighbors (KNN)",
            "type": "Instance-based / Lazy Learner",
            "principle": "Classifies a new data point based on the majority class among its 'K' nearest neighbors in the feature space, measured using distance metrics like Euclidean distance.",
            "hyperparameters": "n_neighbors (k), weights ('uniform', 'distance'), metric ('euclidean', 'manhattan', 'minkowski').",
            "advantages": "No training phase (lazy learner), intuitive, naturally handles multi-class classification, non-parametric.",
            "limitations": "Computationally expensive at prediction time, sensitive to noisy/irrelevant features and scale of features (requires feature scaling)."
        },
        "Decision Tree": {
            "name": "Decision Tree",
            "type": "Tree-based Model",
            "principle": "Recursively splits feature space into decision nodes based on criteria like Gini Impurity or Information Gain (Entropy) to create a flowchart-like tree structure.",
            "hyperparameters": "criterion ('gini', 'entropy'), max_depth, min_samples_split, min_samples_leaf.",
            "advantages": "Easy to visualize and interpret, handles both numerical and categorical features, requires little data preparation.",
            "limitations": "Prone to overfitting (high variance) if tree depth is unconstrained; sensitive to small fluctuations in training data."
        },
        "Support Vector Machine (SVM)": {
            "name": "Support Vector Machine (SVM)",
            "type": "Kernel-based Boundary Classifier",
            "principle": "Finds the optimal decision boundary (hyperplane) that maximizes the margin between different classes in a high-dimensional feature space using support vectors.",
            "hyperparameters": "C (margin penalty), kernel ('linear', 'rbf', 'poly'), gamma (kernel coefficient).",
            "advantages": "Effective in high-dimensional spaces, robust against overfitting in high dimensions, kernel trick captures complex non-linear boundaries.",
            "limitations": "Slow training on large datasets, sensitive to feature scaling, hyperparameter tuning can be tricky, less interpretable."
        },
        "Random Forest": {
            "name": "Random Forest",
            "type": "Ensemble Learning (Bagging)",
            "principle": "Combines multiple decision trees trained on random bootstrap samples of data and random subsets of features, aggregating predictions via majority voting.",
            "hyperparameters": "n_estimators (number of trees), max_depth, min_samples_split, max_features.",
            "advantages": "High accuracy, reduces overfitting risk compared to single decision tree, handles high dimensionality, provides feature importance metrics.",
            "limitations": "Slower prediction time, requires more memory, acts as a 'black-box' (less interpretable than a single decision tree)."
        }
    }
    return info

# Backward compatibility alias
get_model_viva_info = get_model_insights
