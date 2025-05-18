"""
Model training and evaluation page for the ML Analyzer application
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
from datetime import datetime

from assets.styles import HEADING_FONT, SUBHEADING_FONT, NORMAL_FONT
from assets.icons import (
    PLAY_ICON, SAVE_ICON, DOWNLOAD_ICON, 
    CHART_ICON, HOME_ICON, MODEL_ICON
)
from assets.images import MODEL_VISUALIZATION, CONFUSION_MATRIX_TEMPLATE
from gui.components import (
    ScrollableFrame, StatusProgressBar, 
    DataTable, SvgImage, TooltipManager
)
from core.model_trainer import ModelTrainer
from core.evaluation import ModelEvaluator
from core.utils import replace_values_in_svg
from config.settings import (
    CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS,
    CLASSIFICATION_METRICS, REGRESSION_METRICS,
    MODEL_SAVE_DIR, RESULTS_EXPORT_DIR
)

class ModelPage(ttk.Frame):
    """
    Page for training, evaluating and testing machine learning models
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.tooltip_manager = TooltipManager()
        
        # Initialize model related objects
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        
        # Initialize UI components
        self.create_main_ui()
    
    def create_main_ui(self):
        """Create the main UI components"""
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Model Training
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Model Training")
        self.create_training_tab()
        
        # Tab 2: Model Evaluation
        self.evaluation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.evaluation_tab, text="Model Evaluation")
        self.create_evaluation_tab()
        
        # Tab 3: Prediction
        self.prediction_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_tab, text="Prediction")
        self.create_prediction_tab()
        
        # Status bar at the bottom
        self.status_bar = StatusProgressBar(self)
        self.status_bar.pack(fill="x", side="bottom", padx=10, pady=5)
    
    def create_training_tab(self):
        """Create contents of the training tab"""
        frame = ScrollableFrame(self.training_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = frame.scrollable_frame
        
        # Header
        header_frame = ttk.Frame(content)
        header_frame.pack(fill="x", pady=10)
        
        title = ttk.Label(header_frame, text="Model Training", font=HEADING_FONT)
        title.pack(side="left")
        
        # Model visualization
        try:
            viz_image = SvgImage(MODEL_VISUALIZATION, width=600, height=200)
            viz_label = ttk.Label(content, image=viz_image.get())
            viz_label.image = viz_image.get()  # Keep a reference
            viz_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading visualization: {e}")
        
        # Training configuration
        config_frame = ttk.LabelFrame(content, text="Training Configuration", padding=10)
        config_frame.pack(fill="x", pady=10)
        
        # Display selected algorithm and task
        info_frame = ttk.Frame(config_frame)
        info_frame.pack(fill="x", pady=5)
        
        ttk.Label(info_frame, text="Task Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.task_label = ttk.Label(info_frame, text="Not selected", font=NORMAL_FONT)
        self.task_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(info_frame, text="Algorithm:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.algorithm_label = ttk.Label(info_frame, text="Not selected", font=NORMAL_FONT)
        self.algorithm_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(info_frame, text="Target Column:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.target_label = ttk.Label(info_frame, text="Not selected", font=NORMAL_FONT)
        self.target_label.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Model hyperparameters (to be populated based on selected algorithm)
        self.params_frame = ttk.LabelFrame(config_frame, text="Model Hyperparameters", padding=10)
        self.params_frame.pack(fill="x", pady=10)
        
        self.params_placeholder = ttk.Label(
            self.params_frame,
            text="Select an algorithm to configure hyperparameters",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.params_placeholder.pack(pady=10)
        
        # Train-test split configuration
        split_frame = ttk.Frame(config_frame)
        split_frame.pack(fill="x", pady=5)
        
        ttk.Label(split_frame, text="Train-Test Split:").pack(side="left", padx=5)
        self.split_var = tk.DoubleVar(value=0.2)
        split_scale = ttk.Scale(
            split_frame, 
            from_=0.1, 
            to=0.5, 
            variable=self.split_var, 
            orient="horizontal",
            length=200
        )
        split_scale.pack(side="left", padx=5)
        
        self.split_label = ttk.Label(split_frame, text="20%")
        self.split_label.pack(side="left", padx=5)
        
        # Update label when scale changes
        split_scale.bind("<Motion>", self.update_split_label)
        
        # Cross-validation option
        cv_frame = ttk.Frame(config_frame)
        cv_frame.pack(fill="x", pady=5)
        
        self.cv_var = tk.BooleanVar(value=False)
        cv_check = ttk.Checkbutton(
            cv_frame,
            text="Use cross-validation",
            variable=self.cv_var
        )
        cv_check.pack(side="left", padx=5)
        
        self.cv_folds_var = tk.IntVar(value=5)
        self.cv_folds_label = ttk.Label(cv_frame, text="Folds:")
        self.cv_folds_label.pack(side="left", padx=(20, 5))
        self.cv_folds_label.state(["disabled"])
        
        self.cv_folds_spin = ttk.Spinbox(
            cv_frame,
            from_=2,
            to=10,
            textvariable=self.cv_folds_var,
            width=5
        )
        self.cv_folds_spin.pack(side="left", padx=5)
        self.cv_folds_spin.state(["disabled"])
        
        # Enable/disable CV folds based on checkbox
        def update_cv_state():
            state = "!disabled" if self.cv_var.get() else "disabled"
            self.cv_folds_label.state([state])
            self.cv_folds_spin.state([state])
        
        cv_check.config(command=update_cv_state)
        
        # Training controls
        controls_frame = ttk.Frame(content)
        controls_frame.pack(fill="x", pady=10)
        
        # Train button with icon
        try:
            train_icon = SvgImage(PLAY_ICON, width=20, height=20)
            train_button = ttk.Button(
                controls_frame,
                text=" Train Model",
                image=train_icon.get(),
                compound=tk.LEFT,
                command=self.train_model,
                style="Accent.TButton"
            )
            train_button.image = train_icon.get()  # Keep a reference
        except Exception:
            train_button = ttk.Button(
                controls_frame,
                text="Train Model",
                command=self.train_model,
                style="Accent.TButton"
            )
        
        self.tooltip_manager.add_tooltip(train_button, "Train the model with current configuration")
        train_button.pack(side="left", padx=5, pady=10)
        
        # Save model button
        try:
            save_icon = SvgImage(SAVE_ICON, width=20, height=20)
            save_button = ttk.Button(
                controls_frame,
                text=" Save Model",
                image=save_icon.get(),
                compound=tk.LEFT,
                command=self.save_model
            )
            save_button.image = save_icon.get()  # Keep a reference
        except Exception:
            save_button = ttk.Button(
                controls_frame,
                text="Save Model",
                command=self.save_model
            )
        
        self.tooltip_manager.add_tooltip(save_button, "Save the trained model to disk")
        save_button.pack(side="left", padx=5, pady=10)
        
        # Load model button
        load_button = ttk.Button(
            controls_frame,
            text="Load Model",
            command=self.load_model
        )
        self.tooltip_manager.add_tooltip(load_button, "Load a previously saved model")
        load_button.pack(side="left", padx=5, pady=10)
        
        # Back button
        back_button = ttk.Button(
            controls_frame,
            text="Back to Dataset",
            command=lambda: self.controller.show_frame("dataset")
        )
        back_button.pack(side="right", padx=5, pady=10)
        
        # Training results section
        self.results_frame = ttk.LabelFrame(content, text="Training Results", padding=10)
        self.results_frame.pack(fill="both", expand=True, pady=10)
        
        # Placeholder for training results
        self.results_placeholder = ttk.Label(
            self.results_frame,
            text="Train a model to see results",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.results_placeholder.pack(pady=20)
    
    def create_evaluation_tab(self):
        """Create contents of the evaluation tab"""
        frame = ScrollableFrame(self.evaluation_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = frame.scrollable_frame
        
        # Header
        header_frame = ttk.Frame(content)
        header_frame.pack(fill="x", pady=10)
        
        title = ttk.Label(header_frame, text="Model Evaluation", font=HEADING_FONT)
        title.pack(side="left")
        
        # Performance metrics section
        self.metrics_frame = ttk.LabelFrame(content, text="Performance Metrics", padding=10)
        self.metrics_frame.pack(fill="x", pady=10)
        
        # Placeholder for metrics
        self.metrics_placeholder = ttk.Label(
            self.metrics_frame,
            text="Train a model to see evaluation metrics",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.metrics_placeholder.pack(pady=20)
        
        # Confusion matrix (for classification)
        self.confusion_frame = ttk.LabelFrame(content, text="Confusion Matrix", padding=10)
        self.confusion_frame.pack(fill="x", pady=10)
        
        # Placeholder for confusion matrix
        self.confusion_placeholder = ttk.Label(
            self.confusion_frame,
            text="Train a classification model to see confusion matrix",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.confusion_placeholder.pack(pady=20)
        
        # Evaluation plots section
        self.plots_frame = ttk.LabelFrame(content, text="Evaluation Plots", padding=10)
        self.plots_frame.pack(fill="both", expand=True, pady=10)
        
        # Placeholder for plots
        self.plots_placeholder = ttk.Label(
            self.plots_frame,
            text="Train a model to see evaluation plots",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.plots_placeholder.pack(pady=20)
        
        # Export results button
        export_frame = ttk.Frame(content)
        export_frame.pack(fill="x", pady=10)
        
        try:
            export_icon = SvgImage(DOWNLOAD_ICON, width=20, height=20)
            export_button = ttk.Button(
                export_frame,
                text=" Export Results",
                image=export_icon.get(),
                compound=tk.LEFT,
                command=self.export_results
            )
            export_button.image = export_icon.get()  # Keep a reference
        except Exception:
            export_button = ttk.Button(
                export_frame,
                text="Export Results",
                command=self.export_results
            )
        
        self.tooltip_manager.add_tooltip(export_button, "Export evaluation results to CSV")
        export_button.pack(side="left", padx=5, pady=10)
    
    def create_prediction_tab(self):
        """Create contents of the prediction tab"""
        frame = ScrollableFrame(self.prediction_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = frame.scrollable_frame
        
        # Header
        header_frame = ttk.Frame(content)
        header_frame.pack(fill="x", pady=10)
        
        title = ttk.Label(header_frame, text="Make Predictions", font=HEADING_FONT)
        title.pack(side="left")
        
        # Manual prediction section
        self.manual_frame = ttk.LabelFrame(content, text="Manual Prediction", padding=10)
        self.manual_frame.pack(fill="x", pady=10)
        
        # Placeholder for manual prediction inputs
        self.manual_placeholder = ttk.Label(
            self.manual_frame,
            text="Train a model to make predictions",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.manual_placeholder.pack(pady=20)
        
        # Batch prediction section
        batch_frame = ttk.LabelFrame(content, text="Batch Prediction", padding=10)
        batch_frame.pack(fill="x", pady=10)
        
        # File upload for batch prediction
        batch_path_frame = ttk.Frame(batch_frame)
        batch_path_frame.pack(fill="x", pady=5)
        
        self.batch_path_var = tk.StringVar()
        
        ttk.Label(batch_path_frame, text="File:").pack(side="left", padx=5)
        batch_path_entry = ttk.Entry(batch_path_frame, textvariable=self.batch_path_var, width=50)
        batch_path_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        batch_browse_button = ttk.Button(
            batch_path_frame,
            text="Browse",
            command=self.browse_batch_file
        )
        batch_browse_button.pack(side="left", padx=5)
        
        # Batch predict button
        batch_predict_button = ttk.Button(
            batch_frame,
            text="Run Batch Prediction",
            command=self.run_batch_prediction,
            style="Accent.TButton"
        )
        batch_predict_button.pack(pady=10)
        
        # Prediction results section
        self.prediction_results_frame = ttk.LabelFrame(content, text="Prediction Results", padding=10)
        self.prediction_results_frame.pack(fill="both", expand=True, pady=10)
        
        # Placeholder for prediction results
        self.prediction_placeholder = ttk.Label(
            self.prediction_results_frame,
            text="Make a prediction to see results",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.prediction_placeholder.pack(pady=20)
    
    def update_split_label(self, event=None):
        """Update the label showing the train-test split percentage"""
        self.split_label.config(text=f"{int(self.split_var.get() * 100)}%")
    
    def train_model(self):
        """Train a model with the selected configuration"""
        # Check if dataset is loaded and preprocessed
        if self.controller.app_data.get("processed_dataset") is None:
            messagebox.showerror("Error", "Please load and preprocess a dataset first")
            return
        
        # Update labels with current configuration
        self._update_configuration_labels()
        
        # Update hyperparameters based on selected algorithm
        self._update_hyperparameter_inputs()
        
        # Get training parameters
        training_params = self._get_training_parameters()
        
        # Start training animation
        self.status_bar.start_indeterminate("Training model...")
        
        # Use a thread to prevent UI freezing
        threading.Thread(
            target=self._train_model_thread,
            args=(training_params,),
            daemon=True
        ).start()
    
    def _update_configuration_labels(self):
        """Update the configuration display labels"""
        self.task_label.config(text=self.controller.app_data.get("task_type", "Not selected"))
        self.algorithm_label.config(text=self.controller.app_data.get("algorithm", "Not selected"))
        self.target_label.config(text=self.controller.app_data.get("target_column", "Not selected"))
    
    def _update_hyperparameter_inputs(self):
        """Create input widgets for the selected algorithm's hyperparameters"""
        # Clear existing widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        # Get algorithm and task type
        algorithm_name = self.controller.app_data.get("algorithm")
        task_type = self.controller.app_data.get("task_type")
        
        if not algorithm_name or not task_type:
            ttk.Label(
                self.params_frame,
                text="Please select an algorithm on the dataset page",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=10)
            return
        
        # Get algorithm code
        algorithm_code = None
        if task_type == "Classification":
            algorithm_code = CLASSIFICATION_ALGORITHMS.get(algorithm_name)
        else:  # Regression
            algorithm_code = REGRESSION_ALGORITHMS.get(algorithm_name)
        
        if not algorithm_code:
            ttk.Label(
                self.params_frame,
                text=f"Unknown algorithm: {algorithm_name}",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=10)
            return
        
        # Create hyperparameter inputs based on algorithm
        self.param_vars = {}
        
        if algorithm_code == "knn":
            # K-Nearest Neighbors parameters
            ttk.Label(self.params_frame, text="Number of neighbors (k):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["n_neighbors"] = tk.IntVar(value=5)
            ttk.Spinbox(
                self.params_frame,
                from_=1,
                to=50,
                textvariable=self.param_vars["n_neighbors"],
                width=5
            ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            ttk.Label(self.params_frame, text="Weight function:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["weights"] = tk.StringVar(value="uniform")
            ttk.Combobox(
                self.params_frame,
                textvariable=self.param_vars["weights"],
                values=["uniform", "distance"],
                state="readonly",
                width=10
            ).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        elif algorithm_code == "svm":
            # SVM parameters
            ttk.Label(self.params_frame, text="Kernel:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["kernel"] = tk.StringVar(value="rbf")
            ttk.Combobox(
                self.params_frame,
                textvariable=self.param_vars["kernel"],
                values=["linear", "poly", "rbf", "sigmoid"],
                state="readonly",
                width=10
            ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            ttk.Label(self.params_frame, text="C (Regularization):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["C"] = tk.DoubleVar(value=1.0)
            ttk.Spinbox(
                self.params_frame,
                from_=0.1,
                to=100.0,
                increment=0.1,
                textvariable=self.param_vars["C"],
                width=5
            ).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        elif algorithm_code == "decision_tree":
            # Decision Tree parameters
            ttk.Label(self.params_frame, text="Max Depth:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["max_depth"] = tk.IntVar(value=None)
            max_depth_frame = ttk.Frame(self.params_frame)
            max_depth_frame.grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            self.max_depth_none_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                max_depth_frame,
                text="None",
                variable=self.max_depth_none_var,
                command=self._toggle_max_depth
            ).pack(side="left")
            
            self.max_depth_spinbox = ttk.Spinbox(
                max_depth_frame,
                from_=1,
                to=50,
                textvariable=self.param_vars["max_depth"],
                width=5,
                state="disabled"
            )
            self.max_depth_spinbox.pack(side="left", padx=5)
            
            ttk.Label(self.params_frame, text="Min Samples Split:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["min_samples_split"] = tk.IntVar(value=2)
            ttk.Spinbox(
                self.params_frame,
                from_=2,
                to=20,
                textvariable=self.param_vars["min_samples_split"],
                width=5
            ).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        elif algorithm_code == "random_forest":
            # Random Forest parameters
            ttk.Label(self.params_frame, text="Number of Trees:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["n_estimators"] = tk.IntVar(value=100)
            ttk.Spinbox(
                self.params_frame,
                from_=10,
                to=500,
                textvariable=self.param_vars["n_estimators"],
                width=5
            ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            ttk.Label(self.params_frame, text="Max Depth:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["max_depth"] = tk.IntVar(value=None)
            max_depth_frame = ttk.Frame(self.params_frame)
            max_depth_frame.grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
            self.max_depth_none_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                max_depth_frame,
                text="None",
                variable=self.max_depth_none_var,
                command=self._toggle_max_depth
            ).pack(side="left")
            
            self.max_depth_spinbox = ttk.Spinbox(
                max_depth_frame,
                from_=1,
                to=50,
                textvariable=self.param_vars["max_depth"],
                width=5,
                state="disabled"
            )
            self.max_depth_spinbox.pack(side="left", padx=5)
        
        elif algorithm_code == "naive_bayes":
            # Naive Bayes parameters
            ttk.Label(
                self.params_frame,
                text="Gaussian Naive Bayes has no hyperparameters to tune.",
                font=NORMAL_FONT
            ).grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        elif algorithm_code == "linear_regression":
            # Linear Regression parameters
            ttk.Label(
                self.params_frame,
                text="Linear Regression has no hyperparameters to tune.",
                font=NORMAL_FONT
            ).grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        elif algorithm_code == "ridge":
            # Ridge Regression parameters
            ttk.Label(self.params_frame, text="Alpha (Regularization):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["alpha"] = tk.DoubleVar(value=1.0)
            ttk.Spinbox(
                self.params_frame,
                from_=0.01,
                to=10.0,
                increment=0.01,
                textvariable=self.param_vars["alpha"],
                width=5
            ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        elif algorithm_code == "lasso":
            # Lasso Regression parameters
            ttk.Label(self.params_frame, text="Alpha (Regularization):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["alpha"] = tk.DoubleVar(value=1.0)
            ttk.Spinbox(
                self.params_frame,
                from_=0.01,
                to=10.0,
                increment=0.01,
                textvariable=self.param_vars["alpha"],
                width=5
            ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        elif algorithm_code == "svr":
            # SVR parameters
            ttk.Label(self.params_frame, text="Kernel:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["kernel"] = tk.StringVar(value="rbf")
            ttk.Combobox(
                self.params_frame,
                textvariable=self.param_vars["kernel"],
                values=["linear", "poly", "rbf", "sigmoid"],
                state="readonly",
                width=10
            ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            ttk.Label(self.params_frame, text="C (Regularization):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.param_vars["C"] = tk.DoubleVar(value=1.0)
            ttk.Spinbox(
                self.params_frame,
                from_=0.1,
                to=100.0,
                increment=0.1,
                textvariable=self.param_vars["C"],
                width=5
            ).grid(row=1, column=1, sticky="w", padx=5, pady=5)
    
    def _toggle_max_depth(self):
        """Toggle the max_depth parameter based on checkbox"""
        if self.max_depth_none_var.get():
            self.max_depth_spinbox.state(["disabled"])
            self.param_vars["max_depth"].set(None)
        else:
            self.max_depth_spinbox.state(["!disabled"])
            self.param_vars["max_depth"].set(10)  # Default value
    
    def _get_training_parameters(self):
        """Gather all training parameters"""
        # Get algorithm and task type
        algorithm_name = self.controller.app_data.get("algorithm")
        task_type = self.controller.app_data.get("task_type")
        
        # Get algorithm code
        algorithm_code = None
        if task_type == "Classification":
            algorithm_code = CLASSIFICATION_ALGORITHMS.get(algorithm_name)
        else:  # Regression
            algorithm_code = REGRESSION_ALGORITHMS.get(algorithm_name)
        
        # Gather hyperparameters
        hyperparams = {}
        if hasattr(self, 'param_vars'):
            for param, var in self.param_vars.items():
                if param == "max_depth" and hasattr(self, "max_depth_none_var") and self.max_depth_none_var.get():
                    hyperparams[param] = None
                else:
                    value = var.get()
                    # Convert to correct types if needed
                    if isinstance(value, str) and value.lower() == "none":
                        hyperparams[param] = None
                    else:
                        hyperparams[param] = value
        
        # Training configuration
        train_test_split = self.split_var.get()
        use_cv = self.cv_var.get()
        cv_folds = self.cv_folds_var.get() if use_cv else None
        
        return {
            "algorithm": algorithm_code,
            "task_type": task_type,
            "hyperparams": hyperparams,
            "train_test_split": train_test_split,
            "use_cv": use_cv,
            "cv_folds": cv_folds
        }
    
    def _train_model_thread(self, training_params):
        """Thread function for model training"""
        try:
            # Get processed data
            X, y = self.controller.app_data.get("processed_dataset")
            
            # Train the model
            model, train_results = self.model_trainer.train_model(
                X, 
                y, 
                training_params["algorithm"],
                training_params["task_type"],
                hyperparams=training_params["hyperparams"],
                test_size=training_params["train_test_split"],
                cv=training_params["cv_folds"] if training_params["use_cv"] else None
            )
            
            # Store model in application data
            self.controller.app_data["model"] = model
            self.controller.app_data["training_results"] = train_results
            
            # Evaluate the model
            X_test = train_results.get("X_test")
            y_test = train_results.get("y_test")
            
            if X_test is not None and y_test is not None:
                evaluation_results = self.model_evaluator.evaluate_model(
                    model,
                    X_test,
                    y_test,
                    training_params["task_type"]
                )
                
                # Store evaluation results
                self.controller.app_data["evaluation_results"] = evaluation_results
            
            # Update UI on the main thread
            self.after(0, self._update_ui_after_training)
            
        except Exception as e:
            # Update UI on the main thread with error message
            self.after(0, lambda: self._handle_training_error(str(e)))
    
    def _handle_training_error(self, error_message):
        """Handle training errors"""
        self.status_bar.stop_indeterminate("Error during model training")
        messagebox.showerror("Error", f"Failed to train model: {error_message}")
    
    def _update_ui_after_training(self):
        """Update UI components after training is complete"""
        # Stop loading animation
        self.status_bar.stop_indeterminate("Model training complete")
        
        # Update training results tab
        self._update_training_results()
        
        # Update evaluation tab
        self._update_evaluation_results()
        
        # Update prediction tab
        self._update_prediction_tab()
        
        # Switch to evaluation tab
        self.notebook.select(1)  # Index 1 is evaluation tab
    
    def _update_training_results(self):
        """Update the training results display"""
        # Clear existing content
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Get training results
        train_results = self.controller.app_data.get("training_results")
        if not train_results:
            ttk.Label(
                self.results_frame,
                text="No training results available",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=20)
            return
        
        # Create scrollable frame for results
        results_content = ttk.Frame(self.results_frame)
        results_content.pack(fill="both", expand=True)
        
        # Display training information
        info_frame = ttk.Frame(results_content)
        info_frame.pack(fill="x", pady=5)
        
        task_type = self.controller.app_data.get("task_type")
        algorithm = self.controller.app_data.get("algorithm")
        
        # Training summary
        summary_text = f"Model: {algorithm} ({task_type})\n"
        summary_text += f"Training samples: {train_results.get('train_size', 'N/A')}\n"
        summary_text += f"Test samples: {train_results.get('test_size', 'N/A')}\n"
        
        if train_results.get("cv_results"):
            summary_text += f"Cross-validation: {train_results.get('cv_folds', 'N/A')} folds\n"
        
        summary_text += f"Training time: {train_results.get('training_time', 'N/A'):.2f} seconds"
        
        ttk.Label(
            info_frame,
            text=summary_text,
            font=NORMAL_FONT,
            justify=tk.LEFT
        ).pack(anchor="w", padx=10, pady=5)
        
        # Display cross-validation results if available
        if train_results.get("cv_results"):
            cv_frame = ttk.LabelFrame(results_content, text="Cross-Validation Results")
            cv_frame.pack(fill="x", pady=10)
            
            cv_metric = train_results["cv_metric"]
            cv_scores = train_results["cv_scores"]
            
            cv_text = f"Metric: {cv_metric}\n"
            cv_text += f"Mean score: {cv_scores.mean():.4f}\n"
            cv_text += f"Standard deviation: {cv_scores.std():.4f}\n"
            cv_text += "Individual fold scores:\n"
            
            for i, score in enumerate(cv_scores):
                cv_text += f"  Fold {i+1}: {score:.4f}\n"
            
            ttk.Label(
                cv_frame,
                text=cv_text,
                font=NORMAL_FONT,
                justify=tk.LEFT
            ).pack(anchor="w", padx=10, pady=5)
        
        # Display feature importance if available
        if train_results.get("feature_importance") is not None:
            importance_frame = ttk.LabelFrame(results_content, text="Feature Importance")
            importance_frame.pack(fill="x", pady=10)
            
            feature_names = train_results.get("feature_names", [f"Feature {i}" for i in range(len(train_results["feature_importance"]))])
            importances = train_results["feature_importance"]
            
            # Sort by importance
            sorted_idx = importances.argsort()[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importances = importances[sorted_idx]
            
            # Display top 10 features
            top_n = min(10, len(sorted_features))
            
            importance_text = "Top features by importance:\n"
            for i in range(top_n):
                importance_text += f"  {sorted_features[i]}: {sorted_importances[i]:.4f}\n"
            
            ttk.Label(
                importance_frame,
                text=importance_text,
                font=NORMAL_FONT,
                justify=tk.LEFT
            ).pack(anchor="w", padx=10, pady=5)
    
    def _update_evaluation_results(self):
        """Update the evaluation tab with results"""
        # Clear existing content
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        for widget in self.confusion_frame.winfo_children():
            widget.destroy()
        
        for widget in self.plots_frame.winfo_children():
            widget.destroy()
        
        # Get evaluation results
        eval_results = self.controller.app_data.get("evaluation_results")
        if not eval_results:
            ttk.Label(
                self.metrics_frame,
                text="No evaluation results available",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=20)
            
            ttk.Label(
                self.confusion_frame,
                text="No confusion matrix available",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=20)
            
            ttk.Label(
                self.plots_frame,
                text="No evaluation plots available",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=20)
            return
        
        # Get task type
        task_type = self.controller.app_data.get("task_type")
        
        # Update metrics
        metrics_content = ttk.Frame(self.metrics_frame)
        metrics_content.pack(fill="both", expand=True)
        
        if task_type == "Classification":
            # Classification metrics
            metrics_text = (
                f"Accuracy: {eval_results.get('accuracy', 'N/A'):.4f}\n"
                f"Precision: {eval_results.get('precision', 'N/A'):.4f}\n"
                f"Recall: {eval_results.get('recall', 'N/A'):.4f}\n"
                f"F1 Score: {eval_results.get('f1', 'N/A'):.4f}\n"
            )
            
            ttk.Label(
                metrics_content,
                text=metrics_text,
                font=NORMAL_FONT,
                justify=tk.LEFT
            ).pack(anchor="w", padx=10, pady=5)
            
            # Update confusion matrix
            if eval_results.get("confusion_matrix") is not None:
                self._update_confusion_matrix(eval_results["confusion_matrix"])
            else:
                ttk.Label(
                    self.confusion_frame,
                    text="Confusion matrix not available",
                    font=NORMAL_FONT,
                    foreground="gray"
                ).pack(pady=20)
        
        else:  # Regression
            # Regression metrics
            metrics_text = (
                f"Mean Absolute Error (MAE): {eval_results.get('mae', 'N/A'):.4f}\n"
                f"Mean Squared Error (MSE): {eval_results.get('mse', 'N/A'):.4f}\n"
                f"Root Mean Squared Error (RMSE): {eval_results.get('rmse', 'N/A'):.4f}\n"
                f"R² Score: {eval_results.get('r2', 'N/A'):.4f}\n"
            )
            
            ttk.Label(
                metrics_content,
                text=metrics_text,
                font=NORMAL_FONT,
                justify=tk.LEFT
            ).pack(anchor="w", padx=10, pady=5)
            
            # Hide confusion matrix for regression
            ttk.Label(
                self.confusion_frame,
                text="Confusion matrix is only applicable for classification tasks",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=20)
        
        # Update plots
        self._update_evaluation_plots(eval_results, task_type)
    
    def _update_confusion_matrix(self, confusion_matrix):
        """Update the confusion matrix visualization"""
        # Create a frame for the confusion matrix
        cm_frame = ttk.Frame(self.confusion_frame)
        cm_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        try:
            # Try to use the template and replace values
            tn, fp, fn, tp = confusion_matrix.ravel()
            
            # Replace values in SVG template
            cm_svg = replace_values_in_svg(
                CONFUSION_MATRIX_TEMPLATE,
                {
                    "TN": str(tn),
                    "FP": str(fp),
                    "FN": str(fn),
                    "TP": str(tp)
                }
            )
            
            # Create SVG image
            cm_image = SvgImage(cm_svg, width=300, height=300)
            cm_label = ttk.Label(cm_frame, image=cm_image.get())
            cm_label.image = cm_image.get()  # Keep a reference
            cm_label.pack(pady=10)
            
        except Exception as e:
            print(f"Error creating confusion matrix visualization: {e}")
            
            # Fallback to text representation
            cm_text = (
                f"True Negatives (TN): {confusion_matrix[0, 0]}\n"
                f"False Positives (FP): {confusion_matrix[0, 1]}\n"
                f"False Negatives (FN): {confusion_matrix[1, 0]}\n"
                f"True Positives (TP): {confusion_matrix[1, 1]}\n"
            )
            
            ttk.Label(
                cm_frame,
                text=cm_text,
                font=NORMAL_FONT,
                justify=tk.LEFT
            ).pack(anchor="w", padx=10, pady=5)
    
    def _update_evaluation_plots(self, eval_results, task_type):
        """Update the evaluation plots"""
        # Create a frame for plots
        plots_content = ttk.Frame(self.plots_frame)
        plots_content.pack(fill="both", expand=True)
        
        if task_type == "Classification":
            # For classification: ROC curve if available
            if eval_results.get("roc_curve") is not None:
                self._create_roc_curve_plot(plots_content, eval_results["roc_curve"])
            else:
                ttk.Label(
                    plots_content,
                    text="ROC curve not available",
                    font=NORMAL_FONT,
                    foreground="gray"
                ).pack(pady=20)
        
        else:  # Regression
            # For regression: Actual vs Predicted plot
            if eval_results.get("y_true") is not None and eval_results.get("y_pred") is not None:
                self._create_regression_plot(
                    plots_content, 
                    eval_results["y_true"], 
                    eval_results["y_pred"]
                )
            else:
                ttk.Label(
                    plots_content,
                    text="Prediction data not available for plotting",
                    font=NORMAL_FONT,
                    foreground="gray"
                ).pack(pady=20)
    
    def _create_roc_curve_plot(self, parent, roc_data):
        """Create a ROC curve plot"""
        # Unpack ROC data
        fpr, tpr, _ = roc_data
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {eval_results.get("auc", "N/A"):.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _create_regression_plot(self, parent, y_true, y_pred):
        """Create actual vs predicted plot for regression"""
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add R² annotation
        r2 = self.controller.app_data.get("evaluation_results", {}).get("r2", "N/A")
        if r2 != "N/A":
            ax.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                        ha='left', va='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _update_prediction_tab(self):
        """Update the prediction tab with input fields"""
        # Clear existing content
        for widget in self.manual_frame.winfo_children():
            widget.destroy()
        
        # Get model and feature names
        model = self.controller.app_data.get("model")
        if not model:
            ttk.Label(
                self.manual_frame,
                text="No model available for prediction",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=20)
            return
        
        # Get feature names and processed dataset
        X, _ = self.controller.app_data.get("processed_dataset", (None, None))
        if X is None:
            ttk.Label(
                self.manual_frame,
                text="No processed dataset available",
                font=NORMAL_FONT,
                foreground="gray"
            ).pack(pady=20)
            return
        
        # Create scrollable frame if many features
        if X.shape[1] > 10:
            scroll_frame = ScrollableFrame(self.manual_frame)
            scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
            parent_frame = scroll_frame.scrollable_frame
        else:
            parent_frame = self.manual_frame
        
        # Get feature names
        feature_names = self.controller.app_data.get("training_results", {}).get(
            "feature_names", [f"Feature {i+1}" for i in range(X.shape[1])]
        )
        
        # Create inputs for each feature
        entries_frame = ttk.Frame(parent_frame)
        entries_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a grid layout with two columns
        for i in range(len(feature_names)):
            row = i // 2
            col = i % 2 * 2  # 0 or 2
            
            # Label
            ttk.Label(entries_frame, text=f"{feature_names[i]}:").grid(
                row=row, column=col, sticky="e", padx=5, pady=5
            )
            
            # Entry
            entry = ttk.Entry(entries_frame, width=15)
            entry.grid(row=row, column=col+1, sticky="w", padx=5, pady=5)
            
            # Store reference to entry
            if not hasattr(self, "prediction_entries"):
                self.prediction_entries = []
            self.prediction_entries.append((feature_names[i], entry))
        
        # Add predict button
        predict_button = ttk.Button(
            parent_frame,
            text="Make Prediction",
            command=self.make_prediction,
            style="Accent.TButton"
        )
        predict_button.pack(pady=10)
        
        # Clear prediction results
        for widget in self.prediction_results_frame.winfo_children():
            widget.destroy()
        
        ttk.Label(
            self.prediction_results_frame,
            text="Enter values and click 'Make Prediction'",
            font=NORMAL_FONT,
            foreground="gray"
        ).pack(pady=20)
    
    def make_prediction(self):
        """Make a prediction with manually entered values"""
        if not hasattr(self, "prediction_entries") or not self.prediction_entries:
            messagebox.showerror("Error", "No input fields available")
            return
        
        # Get model
        model = self.controller.app_data.get("model")
        if not model:
            messagebox.showerror("Error", "No model available")
            return
        
        # Get values from entries
        try:
            input_values = []
            for feature_name, entry in self.prediction_entries:
                value_str = entry.get().strip()
                if not value_str:
                    messagebox.showerror("Error", f"Please enter a value for {feature_name}")
                    return
                
                try:
                    # Try to convert to float
                    value = float(value_str)
                except ValueError:
                    # If not a number, keep as string
                    value = value_str
                
                input_values.append(value)
            
            # Convert to DataFrame with feature names
            feature_names = [name for name, _ in self.prediction_entries]
            input_df = pd.DataFrame([input_values], columns=feature_names)
            
            # Make prediction
            prediction = self.model_trainer.predict(model, input_df)
            
            # Display prediction result
            self._display_prediction_result(prediction)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")
    
    def _display_prediction_result(self, prediction):
        """Display the prediction result"""
        # Clear existing content
        for widget in self.prediction_results_frame.winfo_children():
            widget.destroy()
        
        # Create result frame
        result_frame = ttk.Frame(self.prediction_results_frame)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Get task type
        task_type = self.controller.app_data.get("task_type")
        
        if task_type == "Classification":
            # For classification, show predicted class and probabilities if available
            if hasattr(prediction, "item"):  # If it's a numpy array with one element
                predicted_class = prediction.item()
            else:
                predicted_class = prediction[0]
            
            result_text = f"Predicted Class: {predicted_class}"
            
            # Check if model has predict_proba method
            if hasattr(self.controller.app_data.get("model"), "predict_proba"):
                try:
                    # Get input data
                    input_values = [entry.get() for _, entry in self.prediction_entries]
                    feature_names = [name for name, _ in self.prediction_entries]
                    input_df = pd.DataFrame([input_values], columns=feature_names)
                    
                    # Get probabilities
                    probas = self.model_trainer.predict_proba(
                        self.controller.app_data.get("model"),
                        input_df
                    )
                    
                    # Add probabilities to result text
                    result_text += "\n\nClass Probabilities:"
                    for i, prob in enumerate(probas[0]):
                        result_text += f"\nClass {i}: {prob:.4f}"
                except Exception as e:
                    print(f"Error getting probabilities: {e}")
        
        else:  # Regression
            # For regression, show predicted value
            if hasattr(prediction, "item"):  # If it's a numpy array with one element
                predicted_value = prediction.item()
            else:
                predicted_value = prediction[0]
            
            result_text = f"Predicted Value: {predicted_value:.4f}"
        
        # Display result
        result_label = ttk.Label(
            result_frame,
            text=result_text,
            font=SUBHEADING_FONT,
            justify=tk.LEFT
        )
        result_label.pack(anchor="w", padx=10, pady=10)
    
    def browse_batch_file(self):
        """Open file browser to select a CSV file for batch prediction"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File for Batch Prediction",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.batch_path_var.set(file_path)
    
    def run_batch_prediction(self):
        """Run batch prediction on a CSV file"""
        file_path = self.batch_path_var.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a file for batch prediction")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File does not exist")
            return
        
        # Get model
        model = self.controller.app_data.get("model")
        if not model:
            messagebox.showerror("Error", "No model available")
            return
        
        # Start prediction animation
        self.status_bar.start_indeterminate("Running batch prediction...")
        
        # Use a thread to prevent UI freezing
        threading.Thread(
            target=self._batch_prediction_thread,
            args=(file_path, model),
            daemon=True
        ).start()
    
    def _batch_prediction_thread(self, file_path, model):
        """Thread function for batch prediction"""
        try:
            # Load the data
            input_data = pd.read_csv(file_path)
            
            # Make prediction
            predictions = self.model_trainer.predict(model, input_data)
            
            # Add predictions to input data
            target_name = self.controller.app_data.get("target_column", "Prediction")
            results_df = input_data.copy()
            results_df[f"Predicted_{target_name}"] = predictions
            
            # Save results
            result_path = os.path.splitext(file_path)[0] + "_predictions.csv"
            results_df.to_csv(result_path, index=False)
            
            # Update UI on the main thread
            self.after(0, lambda: self._handle_batch_prediction_complete(result_path))
            
        except Exception as e:
            # Update UI on the main thread with error message
            self.after(0, lambda: self._handle_batch_prediction_error(str(e)))
    
    def _handle_batch_prediction_error(self, error_message):
        """Handle batch prediction errors"""
        self.status_bar.stop_indeterminate("Error during batch prediction")
        messagebox.showerror("Error", f"Failed to run batch prediction: {error_message}")
    
    def _handle_batch_prediction_complete(self, result_path):
        """Handle successful batch prediction"""
        self.status_bar.stop_indeterminate("Batch prediction complete")
        messagebox.showinfo("Success", f"Batch prediction completed successfully.\nResults saved to: {result_path}")
        
        # Clear existing content in prediction results
        for widget in self.prediction_results_frame.winfo_children():
            widget.destroy()
        
        # Create result frame
        result_frame = ttk.Frame(self.prediction_results_frame)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Display results summary
        ttk.Label(
            result_frame,
            text=f"Batch predictions saved to:\n{result_path}",
            font=NORMAL_FONT,
            justify=tk.LEFT
        ).pack(anchor="w", padx=10, pady=10)
        
        # Add button to open the results
        open_button = ttk.Button(
            result_frame,
            text="Open Results File",
            command=lambda: os.startfile(result_path) if os.name == 'nt' else os.system(f"xdg-open {result_path}")
        )
        open_button.pack(pady=10)
    
    def save_model(self):
        """Save the trained model to a file"""
        # Check if model exists
        model = self.controller.app_data.get("model")
        if not model:
            messagebox.showerror("Error", "No model to save")
            return
        
        # Create save directory if it doesn't exist
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        
        # Generate default filename
        algorithm = self.controller.app_data.get("algorithm", "model").replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{algorithm}_{timestamp}.joblib"
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            initialdir=MODEL_SAVE_DIR,
            initialfile=default_filename,
            defaultextension=".joblib",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Save model metadata along with the model
            metadata = {
                "algorithm": self.controller.app_data.get("algorithm"),
                "task_type": self.controller.app_data.get("task_type"),
                "target_column": self.controller.app_data.get("target_column"),
                "feature_names": self.controller.app_data.get("training_results", {}).get("feature_names"),
                "timestamp": datetime.now().isoformat()
            }
            
            # Create a dictionary with model and metadata
            save_dict = {
                "model": model,
                "metadata": metadata
            }
            
            # Save the model
            joblib.dump(save_dict, file_path)
            
            messagebox.showinfo("Success", f"Model saved successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
    
    def load_model(self):
        """Load a saved model from a file"""
        # Ask for file location
        file_path = filedialog.askopenfilename(
            title="Load Model",
            initialdir=MODEL_SAVE_DIR if os.path.exists(MODEL_SAVE_DIR) else ".",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load the model
            loaded_data = joblib.load(file_path)
            
            # Check if it's a dictionary with model and metadata
            if isinstance(loaded_data, dict) and "model" in loaded_data and "metadata" in loaded_data:
                model = loaded_data["model"]
                metadata = loaded_data["metadata"]
                
                # Update application data with loaded model and metadata
                self.controller.app_data["model"] = model
                self.controller.app_data["algorithm"] = metadata.get("algorithm", "Unknown")
                self.controller.app_data["task_type"] = metadata.get("task_type", "Unknown")
                self.controller.app_data["target_column"] = metadata.get("target_column", "Unknown")
                
                if "training_results" not in self.controller.app_data:
                    self.controller.app_data["training_results"] = {}
                
                self.controller.app_data["training_results"]["feature_names"] = metadata.get("feature_names")
                
                messagebox.showinfo("Success", "Model loaded successfully")
                
                # Update UI
                self._update_configuration_labels()
                self._update_hyperparameter_inputs()
                self._update_prediction_tab()
                
            else:
                # Legacy format - just the model
                model = loaded_data
                self.controller.app_data["model"] = model
                messagebox.showinfo("Success", "Model loaded successfully (no metadata available)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def export_results(self):
        """Export evaluation results to CSV"""
        # Check if results exist
        eval_results = self.controller.app_data.get("evaluation_results")
        if not eval_results:
            messagebox.showerror("Error", "No evaluation results to export")
            return
        
        # Create export directory if it doesn't exist
        os.makedirs(RESULTS_EXPORT_DIR, exist_ok=True)
        
        # Generate default filename
        algorithm = self.controller.app_data.get("algorithm", "model").replace(" ", "_").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"{algorithm}_results_{timestamp}.csv"
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            initialdir=RESULTS_EXPORT_DIR,
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Convert results to DataFrame
            task_type = self.controller.app_data.get("task_type")
            
            if task_type == "Classification":
                results_dict = {
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Value": [
                        eval_results.get("accuracy", "N/A"),
                        eval_results.get("precision", "N/A"),
                        eval_results.get("recall", "N/A"),
                        eval_results.get("f1", "N/A")
                    ]
                }
            else:  # Regression
                results_dict = {
                    "Metric": ["MAE", "MSE", "RMSE", "R²"],
                    "Value": [
                        eval_results.get("mae", "N/A"),
                        eval_results.get("mse", "N/A"),
                        eval_results.get("rmse", "N/A"),
                        eval_results.get("r2", "N/A")
                    ]
                }
            
            results_df = pd.DataFrame(results_dict)
            
            # Save to CSV
            results_df.to_csv(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")
    
    def on_show(self):
        """Called when this frame is shown"""
        self.controller.update_status("Configure and train machine learning models")
        
        # Update configuration labels based on current app data
        self._update_configuration_labels()
        
        # Update hyperparameter inputs
        self._update_hyperparameter_inputs()
        
        # If a model is already loaded, update the UI
        if self.controller.app_data.get("model") is not None:
            # Update evaluation results if available
            if self.controller.app_data.get("evaluation_results") is not None:
                self._update_evaluation_results()
            
            # Update prediction tab
            self._update_prediction_tab()
