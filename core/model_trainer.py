"""
Model training and prediction module for ML Analyzer
"""
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, r2_score

# Import classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Import regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

class ModelTrainer:
    """
    Handles model training, prediction, and related operations
    """
    def __init__(self):
        """Initialize the ModelTrainer"""
        # Dictionary mapping algorithm codes to model classes
        self.classification_models = {
            "knn": KNeighborsClassifier,
            "svm": SVC,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "naive_bayes": GaussianNB
        }
        
        self.regression_models = {
            "linear_regression": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "svr": SVR,
            "decision_tree": DecisionTreeRegressor
        }
        
        # Store the last trained model
        self.model = None
        self.feature_names = None
    
    def train_model(self, X, y, algorithm, task_type, hyperparams=None, 
                   test_size=0.2, random_state=42, cv=None):
        """
        Train a machine learning model
        
        Args:
            X: Features
            y: Target
            algorithm: Algorithm code (e.g., 'knn', 'svm')
            task_type: 'Classification' or 'Regression'
            hyperparams: Dictionary of hyperparameters
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            cv: Number of cross-validation folds (None for no CV)
            
        Returns:
            tuple: (model, results_dict)
        """
        # Default hyperparams if none provided
        if hyperparams is None:
            hyperparams = {}
        
        # Get model class based on algorithm and task type
        model_class = None
        if task_type == 'Classification':
            model_class = self.classification_models.get(algorithm)
        else:  # Regression
            model_class = self.regression_models.get(algorithm)
        
        if model_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Start timing
        start_time = time.time()
        
        # Create model instance with hyperparameters
        model = model_class(**hyperparams)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Prepare results dictionary
        results = {
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }
        
        # Perform cross-validation if requested
        if cv is not None and cv > 1:
            # Choose appropriate scoring metric
            if task_type == 'Classification':
                scoring = make_scorer(accuracy_score)
                metric_name = "accuracy"
            else:  # Regression
                scoring = make_scorer(r2_score)
                metric_name = "r2"
            
            # Run cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
            
            results["cv_folds"] = cv
            results["cv_scores"] = cv_scores
            results["cv_metric"] = metric_name
        
        # Train the model on the full training set
        model.fit(X_train, y_train)
        
        # Record training time
        results["training_time"] = time.time() - start_time
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            results["feature_importance"] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) > 1:
                # For multi-class or multi-output models
                results["feature_importance"] = np.mean(np.abs(model.coef_), axis=0)
            else:
                results["feature_importance"] = np.abs(model.coef_)
        
        # Store feature names if provided
        if hasattr(X, 'columns'):  # For pandas DataFrame
            results["feature_names"] = X.columns.tolist()
        
        # Store trained model
        self.model = model
        
        return model, results
    
    def predict(self, model, X):
        """
        Make predictions with a trained model
        
        Args:
            model: Trained model
            X: Features to predict on
            
        Returns:
            array: Predictions
        """
        if model is None:
            raise ValueError("Model is not trained")
        
        return model.predict(X)
    
    def predict_proba(self, model, X):
        """
        Get probability predictions for classification models
        
        Args:
            model: Trained model
            X: Features to predict on
            
        Returns:
            array: Probability predictions
        """
        if model is None:
            raise ValueError("Model is not trained")
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        return model.predict_proba(X)
    
    def get_feature_importance(self, model, feature_names=None):
        """
        Get feature importance from a trained model if available
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            dict: Feature importances mapped to feature names
        """
        if model is None:
            raise ValueError("Model is not trained")
        
        # Initialize with default feature names if none provided
        if feature_names is None:
            if hasattr(model, 'feature_importances_'):
                feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) > 1:
                    feature_names = [f"Feature {i}" for i in range(model.coef_.shape[1])]
                else:
                    feature_names = [f"Feature {i}" for i in range(len(model.coef_))]
            else:
                return None
        
        # Extract feature importance
        importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_)
        
        if importance is None:
            return None
        
        # Map feature names to importance
        result = dict(zip(feature_names, importance))
        
        # Sort by importance (descending)
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
