from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def evaluate_model(self, model, X_test, y_test, task="classification"):
        y_pred = model.predict(X_test)
        metrics = []
        plot = None

        if task == "classification":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            metrics = [
                f"Accuracy: {acc:.2f}",
                f"Precision: {prec:.2f}",
                f"Recall: {rec:.2f}",
                f"F1 Score: {f1:.2f}"
            ]

            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plot = fig

        elif task == "regression":
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            metrics = [
                f"MAE: {mae:.2f}",
                f"RMSE: {rmse:.2f}",
                f"R² Score: {r2:.2f}"
            ]

            # Plot predicted vs actual
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.set_title("Predicted vs Actual")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            plot = fig

        return metrics, plot
