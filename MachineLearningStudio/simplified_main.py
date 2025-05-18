import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

class MLAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Analyzer")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        # Application state variables
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.preprocessing_pipeline = None
        self.feature_names = None
        
        # Create UI components
        self.create_widgets()
        
        # Status bar at the bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_widgets(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Dataset
        self.dataset_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text="Dataset")
        self.create_dataset_tab()
        
        # Tab 2: Model
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model")
        self.create_model_tab()
        
        # Tab 3: Results
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        self.create_results_tab()
    
    def create_dataset_tab(self):
        # Main frame
        main_frame = ttk.Frame(self.dataset_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding=10)
        file_frame.pack(fill="x", pady=5)
        
        self.file_path = tk.StringVar()
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(file_frame, text="Load Data", command=self.load_data).grid(row=0, column=3, padx=5, pady=5)
        
        # Dataset info section
        self.info_frame = ttk.LabelFrame(main_frame, text="Dataset Information", padding=10)
        self.info_frame.pack(fill="x", pady=5)
        
        self.info_text = tk.Text(self.info_frame, height=5, width=80)
        self.info_text.pack(fill="x", pady=5)
        self.info_text.insert(tk.END, "Load a dataset to see information")
        self.info_text.config(state="disabled")
        
        # Preview section
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill="both", expand=True, pady=5)
        
        # Add a frame for the preview with scrollbars
        preview_container = ttk.Frame(preview_frame)
        preview_container.pack(fill="both", expand=True)
        
        # Scrollbars
        y_scrollbar = ttk.Scrollbar(preview_container)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scrollbar = ttk.Scrollbar(preview_container, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Treeview for data preview
        self.preview_tree = ttk.Treeview(preview_container, 
                                        yscrollcommand=y_scrollbar.set,
                                        xscrollcommand=x_scrollbar.set)
        
        y_scrollbar.config(command=self.preview_tree.yview)
        x_scrollbar.config(command=self.preview_tree.xview)
        
        self.preview_tree.pack(side=tk.LEFT, fill="both", expand=True)
    
    def create_model_tab(self):
        # Main frame
        main_frame = ttk.Frame(self.model_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Model Configuration", padding=10)
        config_frame.pack(fill="x", pady=5)
        
        # Target column selection
        target_frame = ttk.Frame(config_frame)
        target_frame.pack(fill="x", pady=5)
        
        ttk.Label(target_frame, text="Target Column:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(target_frame, textvariable=self.target_var, state="readonly", width=30)
        self.target_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Task type selection
        task_frame = ttk.Frame(config_frame)
        task_frame.pack(fill="x", pady=5)
        
        ttk.Label(task_frame, text="Task Type:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.task_var = tk.StringVar(value="Classification")
        ttk.Radiobutton(task_frame, text="Classification", variable=self.task_var, value="Classification", 
                      command=self.update_algorithm_list).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(task_frame, text="Regression", variable=self.task_var, value="Regression", 
                      command=self.update_algorithm_list).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        # Algorithm selection
        algo_frame = ttk.Frame(config_frame)
        algo_frame.pack(fill="x", pady=5)
        
        ttk.Label(algo_frame, text="Algorithm:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.algorithm_var = tk.StringVar()
        self.algorithm_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var, state="readonly", width=30)
        self.algorithm_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Train-test split
        split_frame = ttk.Frame(config_frame)
        split_frame.pack(fill="x", pady=5)
        
        ttk.Label(split_frame, text="Test Size:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_scale = ttk.Scale(split_frame, from_=0.1, to=0.5, variable=self.test_size_var, 
                                  orient="horizontal", length=200)
        test_size_scale.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.test_size_label = ttk.Label(split_frame, text="20%")
        self.test_size_label.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        test_size_scale.bind("<Motion>", self.update_test_size_label)
        
        # Preprocessing options
        preprocess_frame = ttk.LabelFrame(main_frame, text="Preprocessing Options", padding=10)
        preprocess_frame.pack(fill="x", pady=5)
        
        # Missing values
        ttk.Label(preprocess_frame, text="Missing Values:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.missing_var = tk.StringVar(value="mean")
        ttk.Radiobutton(preprocess_frame, text="Replace with Mean/Mode", variable=self.missing_var, 
                      value="mean").grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(preprocess_frame, text="Replace with Zeros", variable=self.missing_var, 
                      value="zeros").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        ttk.Radiobutton(preprocess_frame, text="Drop Rows", variable=self.missing_var, 
                      value="drop").grid(row=0, column=3, sticky="w", padx=5, pady=5)
        
        # Normalization
        ttk.Label(preprocess_frame, text="Normalization:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.scaling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocess_frame, text="Standardize Numerical Features", 
                       variable=self.scaling_var).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Training button
        train_button = ttk.Button(main_frame, text="Train Model", command=self.train_model)
        train_button.pack(pady=10)
    
    def create_results_tab(self):
        # Main frame
        main_frame = ttk.Frame(self.results_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Metrics section
        self.metrics_frame = ttk.LabelFrame(main_frame, text="Performance Metrics", padding=10)
        self.metrics_frame.pack(fill="x", pady=5)
        
        self.metrics_text = tk.Text(self.metrics_frame, height=8, width=80)
        self.metrics_text.pack(fill="x", pady=5)
        self.metrics_text.insert(tk.END, "Train a model to see metrics")
        self.metrics_text.config(state="disabled")
        
        # Visualization section
        self.viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding=10)
        self.viz_frame.pack(fill="both", expand=True, pady=5)
        
        # Placeholder for visualization
        self.viz_placeholder = ttk.Label(self.viz_frame, text="Train a model to see visualization")
        self.viz_placeholder.pack(pady=20)
        
        # Prediction section
        prediction_frame = ttk.LabelFrame(main_frame, text="Make Predictions", padding=10)
        prediction_frame.pack(fill="x", pady=5)
        
        # We'll populate this dynamically after a model is trained
        self.prediction_content = ttk.Frame(prediction_frame)
        self.prediction_content.pack(fill="x", pady=5)
        
        self.prediction_placeholder = ttk.Label(self.prediction_content, text="Train a model to make predictions")
        self.prediction_placeholder.pack(pady=20)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.file_path.set(file_path)
    
    def load_data(self):
        file_path = self.file_path.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a dataset file")
            return
        
        try:
            self.update_status("Loading dataset...")
            
            # Load data
            self.data = pd.read_csv(file_path)
            
            # Update the info section
            self.update_dataset_info()
            
            # Update the preview
            self.update_preview()
            
            # Update target column options
            self.target_combo['values'] = self.data.columns.tolist()
            if len(self.data.columns) > 0:
                self.target_var.set(self.data.columns[-1])  # Default to last column
            
            # Update algorithm list based on default task
            self.update_algorithm_list()
            
            self.update_status(f"Loaded dataset with {len(self.data)} rows and {len(self.data.columns)} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.update_status("Error loading dataset")
    
    def update_dataset_info(self):
        if self.data is None:
            return
        
        # Generate dataset information
        info = f"Rows: {len(self.data)}\n"
        info += f"Columns: {len(self.data.columns)}\n"
        info += f"Missing values: {self.data.isna().sum().sum()}\n"
        
        # Calculate memory usage
        memory_usage = self.data.memory_usage(deep=True).sum()
        if memory_usage < 1024:
            memory_info = f"{memory_usage} bytes"
        elif memory_usage < 1024 * 1024:
            memory_info = f"{memory_usage/1024:.2f} KB"
        else:
            memory_info = f"{memory_usage/(1024*1024):.2f} MB"
        
        info += f"Memory usage: {memory_info}"
        
        # Update info text
        self.info_text.config(state="normal")
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state="disabled")
    
    def update_preview(self):
        if self.data is None:
            return
        
        # Clear existing data
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        # Configure columns
        self.preview_tree["columns"] = self.data.columns.tolist()
        self.preview_tree["show"] = "headings"  # Hide the first empty column
        
        # Set column headings
        for col in self.data.columns:
            self.preview_tree.heading(col, text=col)
            # Calculate column width based on content
            max_width = max(len(str(col)), 
                          max([len(str(row[col])) for _, row in self.data.head(10).iterrows()]) if len(self.data) > 0 else 0)
            self.preview_tree.column(col, width=min(max_width * 10, 150), minwidth=50)
        
        # Insert data
        for i, row in self.data.head(10).iterrows():
            values = [str(row[col]) for col in self.data.columns]
            self.preview_tree.insert("", tk.END, text=f"Row {i+1}", values=values)
    
    def update_algorithm_list(self):
        task_type = self.task_var.get()
        
        if task_type == "Classification":
            algorithms = {
                "K-Nearest Neighbors": "knn",
                "Support Vector Machine": "svm",
                "Decision Tree": "decision_tree"
            }
        else:  # Regression
            algorithms = {
                "Linear Regression": "linear_regression",
                "Decision Tree Regressor": "decision_tree"
            }
        
        # Update the algorithm combobox
        self.algorithm_combo['values'] = list(algorithms.keys())
        if len(algorithms) > 0:
            self.algorithm_combo.current(0)
    
    def update_test_size_label(self, event=None):
        self.test_size_label.config(text=f"{int(self.test_size_var.get() * 100)}%")
    
    def update_status(self, message):
        self.status_var.set(message)
        self.update_idletasks()
    
    def train_model(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return
        
        # Get configuration
        target_column = self.target_var.get()
        task_type = self.task_var.get()
        algorithm = self.algorithm_var.get()
        test_size = self.test_size_var.get()
        
        if not target_column:
            messagebox.showerror("Error", "Please select a target column")
            return
        
        try:
            self.update_status("Preprocessing data...")
            
            # Separate features and target
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Identify column types
            numerical_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Preprocessing pipelines
            preprocessor_steps = []
            
            # Numerical preprocessing
            if numerical_cols:
                if self.missing_var.get() == "mean":
                    num_imputer = SimpleImputer(strategy='mean')
                elif self.missing_var.get() == "zeros":
                    num_imputer = SimpleImputer(strategy='constant', fill_value=0)
                else:  # drop
                    X = X.dropna()
                    y = y[X.index]
                    num_imputer = SimpleImputer(strategy='mean')  # Fallback
                
                if self.scaling_var.get():
                    num_pipeline = Pipeline([
                        ('imputer', num_imputer),
                        ('scaler', StandardScaler())
                    ])
                else:
                    num_pipeline = Pipeline([
                        ('imputer', num_imputer)
                    ])
                
                preprocessor_steps.append(('num', num_pipeline, numerical_cols))
            
            # Categorical preprocessing
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                
                cat_pipeline = Pipeline([
                    ('imputer', cat_imputer),
                    ('encoder', cat_encoder)
                ])
                
                preprocessor_steps.append(('cat', cat_pipeline, categorical_cols))
            
            # Create preprocessor
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='drop'
            )
            
            # Apply preprocessing
            X_processed = self.preprocessing_pipeline.fit_transform(X)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42
            )
            
            self.update_status("Training model...")
            
            # Train model
            if task_type == "Classification":
                if algorithm == "K-Nearest Neighbors":
                    self.model = KNeighborsClassifier(n_neighbors=5)
                elif algorithm == "Support Vector Machine":
                    self.model = SVC(probability=True)
                elif algorithm == "Decision Tree":
                    self.model = DecisionTreeClassifier()
            else:  # Regression
                if algorithm == "Linear Regression":
                    self.model = LinearRegression()
                elif algorithm == "Decision Tree Regressor":
                    self.model = DecisionTreeRegressor()
            
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            # Evaluate the model
            self.evaluate_model(task_type)
            
            # Create the prediction interface
            self.create_prediction_interface(X.columns, task_type)
            
            self.update_status(f"Trained {algorithm} model")
            
            # Switch to results tab
            self.notebook.select(2)  # Index 2 is Results tab
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during model training: {str(e)}")
            self.update_status("Error training model")
    
    def evaluate_model(self, task_type):
        if self.model is None or self.X_test is None or self.y_test is None:
            return
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Update metrics text
        self.metrics_text.config(state="normal")
        self.metrics_text.delete(1.0, tk.END)
        
        if task_type == "Classification":
            # Classification metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            metrics_text = f"Accuracy: {accuracy:.4f}\n"
            metrics_text += f"Precision: {precision:.4f}\n"
            metrics_text += f"Recall: {recall:.4f}\n"
            metrics_text += f"F1 Score: {f1:.4f}\n\n"
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(cm)
            
            self.metrics_text.insert(tk.END, metrics_text)
            
            # Visualization: Confusion matrix
            if self.viz_placeholder is not None:
                self.viz_placeholder.destroy()
                self.viz_placeholder = None
            
            fig, ax = plt.subplots(figsize=(6, 5))
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            fig.colorbar(cax)
            
            # Add labels
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')
            
            # Add the canvas to the visualization frame
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        else:  # Regression
            # Regression metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            metrics_text = f"Mean Absolute Error (MAE): {mae:.4f}\n"
            metrics_text += f"Mean Squared Error (MSE): {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
            self.metrics_text.insert(tk.END, metrics_text)
            
            # Visualization: Predicted vs Actual
            if self.viz_placeholder is not None:
                self.viz_placeholder.destroy()
                self.viz_placeholder = None
            
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--')
            
            # Add labels
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Predicted vs Actual')
            
            # Add the canvas to the visualization frame
            canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.metrics_text.config(state="disabled")
    
    def create_prediction_interface(self, feature_columns, task_type):
        # Clear previous content
        for widget in self.prediction_content.winfo_children():
            widget.destroy()
        
        ttk.Label(self.prediction_content, text="Enter values for prediction:").pack(anchor="w", pady=(0, 5))
        
        # Create frame for input fields
        input_frame = ttk.Frame(self.prediction_content)
        input_frame.pack(fill="x", pady=5)
        
        # Add input fields for each feature
        self.prediction_vars = {}
        for i, col in enumerate(feature_columns):
            ttk.Label(input_frame, text=f"{col}:").grid(row=i//3, column=(i%3)*2, sticky="w", padx=5, pady=2)
            self.prediction_vars[col] = tk.StringVar()
            ttk.Entry(input_frame, textvariable=self.prediction_vars[col], width=15).grid(
                row=i//3, column=(i%3)*2+1, sticky="w", padx=5, pady=2)
        
        # Add predict button
        predict_button = ttk.Button(self.prediction_content, text="Predict", command=self.make_prediction)
        predict_button.pack(pady=10)
        
        # Add result label
        self.prediction_result_var = tk.StringVar(value="Prediction will appear here")
        result_label = ttk.Label(self.prediction_content, textvariable=self.prediction_result_var, 
                               font=("Arial", 12, "bold"))
        result_label.pack(pady=5)
    
    def make_prediction(self):
        if self.model is None or self.preprocessing_pipeline is None:
            messagebox.showerror("Error", "No trained model available")
            return
        
        try:
            # Create a dataframe from input values
            input_data = {}
            for col, var in self.prediction_vars.items():
                value = var.get()
                # Try to convert to numeric if possible
                try:
                    value = float(value)
                    if value.is_integer():
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                input_data[col] = [value]
            
            input_df = pd.DataFrame(input_data)
            
            # Apply preprocessing
            processed_input = self.preprocessing_pipeline.transform(input_df)
            
            # Make prediction
            prediction = self.model.predict(processed_input)[0]
            
            # Format and display prediction
            if isinstance(prediction, (np.integer, np.floating)):
                formatted_prediction = f"{prediction:.4f}"
            else:
                formatted_prediction = str(prediction)
            
            self.prediction_result_var.set(f"Prediction: {formatted_prediction}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")
            self.prediction_result_var.set("Error making prediction")

if __name__ == "__main__":
    app = MLAnalyzerApp()
    app.mainloop()