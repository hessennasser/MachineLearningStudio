"""
Configuration settings for the ML Analyzer application
"""

# Application settings
APP_TITLE = "ML Analyzer"
APP_VERSION = "2.0"
DEFAULT_GEOMETRY = "1024x768"

# UI settings
PADX = 10
PADY = 10
BUTTON_PADX = 5
BUTTON_PADY = 5

# Dataset preview settings
PREVIEW_ROWS = 10
MAX_COLUMNS_DISPLAY = 15

# Supported algorithms
CLASSIFICATION_ALGORITHMS = {
    "K-Nearest Neighbors": "knn",
    "Support Vector Machine": "svm",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Naive Bayes": "naive_bayes"
}

REGRESSION_ALGORITHMS = {
    "Linear Regression": "linear_regression",
    "Ridge Regression": "ridge",
    "Lasso Regression": "lasso",
    "Support Vector Regressor": "svr",
    "Decision Tree Regressor": "decision_tree"
}

# Evaluation metrics
CLASSIFICATION_METRICS = [
    "accuracy", "precision", "recall", "f1"
]

REGRESSION_METRICS = [
    "mae", "mse", "rmse", "r2"
]

# File paths
MODEL_SAVE_DIR = "./saved_models"
RESULTS_EXPORT_DIR = "./exports"

# Default column types
DEFAULT_DATETIME_COLUMNS = ["date", "time", "datetime", "timestamp"]
