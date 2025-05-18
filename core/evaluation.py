"""
Model evaluation module for ML Analyzer
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc
)

class ModelEvaluator:
    """
    Evaluates machine learning models and provides performance metrics
    """
    def __init__(self):
        """Initialize the ModelEvaluator"""
        pass
    
    def evaluate_model(self, model, X_test, y_test, task_type):
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            task_type: 'Classification' or 'Regression'
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if model is None:
            raise ValueError("Model is not trained")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics based on task type
        if task_type == 'Classification':
            return self._evaluate_classification(model, X_test, y_test, y_pred)
        else:  # Regression
            return self._evaluate_regression(y_test, y_pred)
    
    def _evaluate_classification(self, model, X_test, y_test, y_pred):
        """
        Evaluate a classification model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            y_pred: Model predictions
            
        Returns:
            dict: Classification metrics
        """
        results = {
            "y_true": y_test,
            "y_pred": y_pred
        }
        
        # Calculate basic metrics
        results["accuracy"] = accuracy_score(y_test, y_pred)
        
        # Handle multi-class classification
        if len(np.unique(y_test)) > 2:
            # Multi-class metrics
            results["precision"] = precision_score(y_test, y_pred, average='weighted')
            results["recall"] = recall_score(y_test, y_pred, average='weighted')
            results["f1"] = f1_score(y_test, y_pred, average='weighted')
            
            # Confusion matrix
            results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
            
            # ROC curve (only for binary classification)
            results["roc_curve"] = None
            results["auc"] = None
        else:
            # Binary classification metrics
            results["precision"] = precision_score(y_test, y_pred)
            results["recall"] = recall_score(y_test, y_pred)
            results["f1"] = f1_score(y_test, y_pred)
            
            # Confusion matrix
            results["confusion_matrix"] = confusion_matrix(y_test, y_pred)
            
            # ROC curve
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_proba)
                results["roc_curve"] = (fpr, tpr, thresholds)
                results["auc"] = auc(fpr, tpr)
        
        return results
    
    def _evaluate_regression(self, y_test, y_pred):
        """
        Evaluate a regression model
        
        Args:
            y_test: Test targets
            y_pred: Model predictions
            
        Returns:
            dict: Regression metrics
        """
        results = {
            "y_true": y_test,
            "y_pred": y_pred
        }
        
        # Calculate metrics
        results["mae"] = mean_absolute_error(y_test, y_pred)
        results["mse"] = mean_squared_error(y_test, y_pred)
        results["rmse"] = np.sqrt(results["mse"])
        results["r2"] = r2_score(y_test, y_pred)
        
        return results
    
    def get_confusion_matrix_text(self, confusion_matrix):
        """
        Get a text representation of a confusion matrix
        
        Args:
            confusion_matrix: NumPy array containing the confusion matrix
            
        Returns:
            str: Text representation
        """
        if confusion_matrix is None:
            return "No confusion matrix available"
        
        if confusion_matrix.shape == (2, 2):
            # Binary classification
            tn, fp, fn, tp = confusion_matrix.ravel()
            return (
                f"True Negatives (TN): {tn}\n"
                f"False Positives (FP): {fp}\n"
                f"False Negatives (FN): {fn}\n"
                f"True Positives (TP): {tp}"
            )
        else:
            # Multi-class classification
            text = "Confusion Matrix:\n"
            
            # Create a header with class numbers
            header = "   |" + "|".join(f" {i} " for i in range(confusion_matrix.shape[0])) + "|"
            text += header + "\n"
            text += "-" * len(header) + "\n"
            
            # Add each row
            for i, row in enumerate(confusion_matrix):
                row_text = f" {i} |" + "|".join(f" {val} " for val in row) + "|"
                text += row_text + "\n"
            
            return text
    
    def get_classification_report_text(self, precision, recall, f1, accuracy):
        """
        Get a text representation of classification metrics
        
        Args:
            precision: Precision score
            recall: Recall score
            f1: F1 score
            accuracy: Accuracy score
            
        Returns:
            str: Text representation
        """
        return (
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}"
        )
    
    def get_regression_report_text(self, mae, mse, rmse, r2):
        """
        Get a text representation of regression metrics
        
        Args:
            mae: Mean absolute error
            mse: Mean squared error
            rmse: Root mean squared error
            r2: R-squared score
            
        Returns:
            str: Text representation
        """
        return (
            f"Mean Absolute Error (MAE): {mae:.4f}\n"
            f"Mean Squared Error (MSE): {mse:.4f}\n"
            f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
            f"R² Score: {r2:.4f}"
        )
