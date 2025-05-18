"""
Dataset upload and configuration page for the ML Analyzer application
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import pandas as pd
from PIL import Image, ImageTk

from assets.styles import HEADING_FONT, SUBHEADING_FONT, NORMAL_FONT
from assets.icons import UPLOAD_ICON, NEXT_ICON, BACK_ICON, DATA_ICON
from assets.images import DATASET_VISUALIZATION, EDA_VISUALIZATION
from gui.components import (
    ScrollableFrame, StatusProgressBar, 
    DataTable, SvgImage, TooltipManager
)
from core.data_handler import DataHandler
from config.settings import (
    CLASSIFICATION_ALGORITHMS, REGRESSION_ALGORITHMS,
    PREVIEW_ROWS, DEFAULT_DATETIME_COLUMNS
)

class DatasetPage(ttk.Frame):
    """
    Page for uploading and configuring dataset settings
    """
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.data_handler = DataHandler()
        self.tooltip_manager = TooltipManager()
        
        # Initialize variables
        self.dataset_path = tk.StringVar()
        self.task_type = tk.StringVar(value="Classification")
        self.algorithm = tk.StringVar()
        self.target_column = tk.StringVar()
        
        # Initialize UI components
        self.create_main_ui()
    
    def create_main_ui(self):
        """Create the main UI components"""
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tab 1: Dataset Upload
        self.upload_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.upload_tab, text="Dataset Upload")
        self.create_upload_tab()
        
        # Tab 2: Data Preprocessing
        self.preprocess_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.preprocess_tab, text="Data Preprocessing")
        self.create_preprocess_tab()
        
        # Tab 3: Exploratory Analysis
        self.eda_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eda_tab, text="Exploratory Analysis")
        self.create_eda_tab()
        
        # Status bar at the bottom
        self.status_bar = StatusProgressBar(self)
        self.status_bar.pack(fill="x", side="bottom", padx=10, pady=5)
        
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def create_upload_tab(self):
        """Create contents of the upload tab"""
        frame = ScrollableFrame(self.upload_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = frame.scrollable_frame
        
        # Header
        header_frame = ttk.Frame(content)
        header_frame.pack(fill="x", pady=10)
        
        title = ttk.Label(header_frame, text="Dataset Configuration", font=HEADING_FONT)
        title.pack(side="left")
        
        # Dataset upload section
        upload_frame = ttk.LabelFrame(content, text="Dataset Upload", padding=10)
        upload_frame.pack(fill="x", pady=10)
        
        # File path display and browse button
        path_frame = ttk.Frame(upload_frame)
        path_frame.pack(fill="x", pady=5)
        
        path_label = ttk.Label(path_frame, text="File Path:")
        path_label.pack(side="left", padx=5)
        
        path_entry = ttk.Entry(path_frame, textvariable=self.dataset_path, width=50)
        path_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        browse_button = ttk.Button(
            path_frame, 
            text="Browse", 
            command=self.browse_file
        )
        self.tooltip_manager.add_tooltip(browse_button, "Select a CSV file to upload")
        browse_button.pack(side="left", padx=5)
        
        # Upload button with icon
        upload_button_frame = ttk.Frame(upload_frame)
        upload_button_frame.pack(pady=10)
        
        try:
            upload_icon = SvgImage(UPLOAD_ICON, width=20, height=20)
            upload_button = ttk.Button(
                upload_button_frame,
                text=" Upload Dataset",
                image=upload_icon.get(),
                compound=tk.LEFT,
                command=self.load_dataset
            )
            upload_button.image = upload_icon.get()  # Keep a reference
        except:
            # Fallback without icon if image loading fails
            upload_button = ttk.Button(
                upload_button_frame,
                text="Upload Dataset",
                command=self.load_dataset
            )
        
        self.tooltip_manager.add_tooltip(upload_button, "Load the selected dataset")
        upload_button.pack(pady=5)
        
        # Dataset preview section
        self.preview_frame = ttk.LabelFrame(content, text="Dataset Preview", padding=10)
        self.preview_frame.pack(fill="both", expand=True, pady=10)
        
        # Initially show a placeholder message
        self.preview_placeholder = ttk.Label(
            self.preview_frame,
            text="Upload a dataset to see preview",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.preview_placeholder.pack(pady=20)
        
        # Data table will replace this when data is loaded
        self.data_table = None
        
        # Task configuration section
        config_frame = ttk.LabelFrame(content, text="Model Configuration", padding=10)
        config_frame.pack(fill="x", pady=10)
        
        # Task type selection (Classification/Regression)
        task_frame = ttk.Frame(config_frame)
        task_frame.pack(fill="x", pady=5)
        
        task_label = ttk.Label(task_frame, text="Task Type:")
        task_label.pack(side="left", padx=5)
        
        task_rb_frame = ttk.Frame(task_frame)
        task_rb_frame.pack(side="left")
        
        classification_rb = ttk.Radiobutton(
            task_rb_frame, 
            text="Classification", 
            variable=self.task_type,
            value="Classification",
            command=self.update_algorithm_list
        )
        classification_rb.pack(side="left", padx=10)
        self.tooltip_manager.add_tooltip(
            classification_rb, 
            "Use when predicting categories or classes"
        )
        
        regression_rb = ttk.Radiobutton(
            task_rb_frame, 
            text="Regression", 
            variable=self.task_type,
            value="Regression",
            command=self.update_algorithm_list
        )
        regression_rb.pack(side="left", padx=10)
        self.tooltip_manager.add_tooltip(
            regression_rb, 
            "Use when predicting continuous numerical values"
        )
        
        # Algorithm selection
        algo_frame = ttk.Frame(config_frame)
        algo_frame.pack(fill="x", pady=5)
        
        algo_label = ttk.Label(algo_frame, text="Algorithm:")
        algo_label.pack(side="left", padx=5)
        
        self.algo_combo = ttk.Combobox(
            algo_frame, 
            textvariable=self.algorithm,
            state="readonly"
        )
        self.algo_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Target column selection
        target_frame = ttk.Frame(config_frame)
        target_frame.pack(fill="x", pady=5)
        
        target_label = ttk.Label(target_frame, text="Target Column:")
        target_label.pack(side="left", padx=5)
        
        self.target_combo = ttk.Combobox(
            target_frame, 
            textvariable=self.target_column,
            state="readonly"
        )
        self.target_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Initialize the algorithm combobox
        self.update_algorithm_list()
        
        # Navigation buttons
        nav_frame = ttk.Frame(content)
        nav_frame.pack(fill="x", pady=20)
        
        back_button = ttk.Button(
            nav_frame,
            text="Back to Welcome",
            command=lambda: self.controller.show_frame("welcome")
        )
        back_button.pack(side="left", padx=5)
        
        next_button = ttk.Button(
            nav_frame,
            text="Continue to Model Training",
            command=self.proceed_to_model,
            style="Accent.TButton"
        )
        next_button.pack(side="right", padx=5)
        self.tooltip_manager.add_tooltip(
            next_button, 
            "Configure model parameters and start training"
        )
    
    def create_preprocess_tab(self):
        """Create contents of the preprocessing tab"""
        frame = ScrollableFrame(self.preprocess_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = frame.scrollable_frame
        
        # Header
        header_frame = ttk.Frame(content)
        header_frame.pack(fill="x", pady=10)
        
        title = ttk.Label(header_frame, text="Data Preprocessing", font=HEADING_FONT)
        title.pack(side="left")
        
        # Data visualization
        try:
            viz_image = SvgImage(DATASET_VISUALIZATION, width=600, height=200)
            viz_label = ttk.Label(content, image=viz_image.get())
            viz_label.image = viz_image.get()  # Keep a reference
            viz_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading visualization: {e}")
        
        # Data summary section
        self.summary_frame = ttk.LabelFrame(content, text="Data Summary", padding=10)
        self.summary_frame.pack(fill="both", expand=True, pady=10)
        
        # Initially show a placeholder message
        self.summary_placeholder = ttk.Label(
            self.summary_frame,
            text="Upload a dataset to see summary",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.summary_placeholder.pack(pady=20)
        
        # Preprocessing options section
        self.preprocess_options_frame = ttk.LabelFrame(content, text="Preprocessing Options", padding=10)
        self.preprocess_options_frame.pack(fill="x", pady=10)
        
        # Initially show a placeholder message
        self.preprocess_placeholder = ttk.Label(
            self.preprocess_options_frame,
            text="Upload a dataset to configure preprocessing",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.preprocess_placeholder.pack(pady=20)
    
    def create_eda_tab(self):
        """Create contents of the exploratory analysis tab"""
        frame = ScrollableFrame(self.eda_tab)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        content = frame.scrollable_frame
        
        # Header
        header_frame = ttk.Frame(content)
        header_frame.pack(fill="x", pady=10)
        
        title = ttk.Label(header_frame, text="Exploratory Data Analysis", font=HEADING_FONT)
        title.pack(side="left")
        
        # EDA visualization
        try:
            viz_image = SvgImage(EDA_VISUALIZATION, width=600, height=300)
            viz_label = ttk.Label(content, image=viz_image.get())
            viz_label.image = viz_image.get()  # Keep a reference
            viz_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading visualization: {e}")
        
        # EDA content section (to be populated when data is loaded)
        self.eda_content_frame = ttk.Frame(content)
        self.eda_content_frame.pack(fill="both", expand=True, pady=10)
        
        # Initially show a placeholder message
        self.eda_placeholder = ttk.Label(
            self.eda_content_frame,
            text="Upload a dataset to see exploratory analysis",
            font=NORMAL_FONT,
            foreground="gray"
        )
        self.eda_placeholder.pack(pady=20)
    
    def browse_file(self):
        """Open file browser to select a CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.dataset_path.set(file_path)
    
    def load_dataset(self):
        """Load the selected dataset and display preview"""
        file_path = self.dataset_path.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a dataset file")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File does not exist")
            return
        
        # Start loading animation
        self.status_bar.start_indeterminate("Loading dataset...")
        
        # Use a thread to prevent UI freezing
        threading.Thread(target=self._load_dataset_thread, args=(file_path,), daemon=True).start()
    
    def _load_dataset_thread(self, file_path):
        """Thread function for loading dataset"""
        try:
            # Load the data using DataHandler
            success = self.data_handler.load_dataset(file_path)
            
            if not success:
                # Update UI on the main thread
                self.after(0, lambda: self._handle_dataset_load_error("Failed to load dataset"))
                return
            
            # Update UI on the main thread
            self.after(0, self._update_ui_after_dataset_load)
            
        except Exception as e:
            # Update UI on the main thread with error message
            self.after(0, lambda: self._handle_dataset_load_error(str(e)))
    
    def _handle_dataset_load_error(self, error_message):
        """Handle dataset loading errors"""
        self.status_bar.stop_indeterminate("Error loading dataset")
        messagebox.showerror("Error", f"Failed to load dataset: {error_message}")
    
    def _update_ui_after_dataset_load(self):
        """Update UI components after dataset is loaded successfully"""
        # Update application data
        self.controller.app_data["dataset"] = self.data_handler.data
        self.controller.app_data["dataset_path"] = self.dataset_path.get()
        
        # Update target column dropdown
        self.target_combo['values'] = self.data_handler.get_column_names()
        if self.target_combo['values']:
            # Default to the last column as target
            self.target_column.set(self.target_combo['values'][-1])
        
        # Remove placeholders and add dataset preview
        if hasattr(self, 'preview_placeholder') and self.preview_placeholder.winfo_exists():
            self.preview_placeholder.destroy()
        
        # Create data table if it doesn't exist
        if self.data_table is None:
            self.data_table = DataTable(self.preview_frame)
            self.data_table.pack(fill="both", expand=True)
        
        # Update the data table with preview rows
        preview_data = self.data_handler.get_preview(PREVIEW_ROWS)
        if preview_data is not None:
            headers = self.data_handler.get_column_names()
            rows = preview_data.values.tolist()
            self.data_table.set_data(headers, rows)
        
        # Update the preprocessing tab
        self._update_preprocessing_tab()
        
        # Update the EDA tab
        self._update_eda_tab()
        
        # Stop loading animation
        self.status_bar.stop_indeterminate("Dataset loaded successfully")
        self.controller.update_status(f"Loaded dataset with {len(self.data_handler.data)} rows and {len(self.data_handler.get_column_names())} columns")
    
    def _update_preprocessing_tab(self):
        """Update preprocessing tab with dataset information"""
        # Remove placeholders
        if hasattr(self, 'summary_placeholder') and self.summary_placeholder.winfo_exists():
            self.summary_placeholder.destroy()
        
        if hasattr(self, 'preprocess_placeholder') and self.preprocess_placeholder.winfo_exists():
            self.preprocess_placeholder.destroy()
        
        # Create data summary
        summary_content = ttk.Frame(self.summary_frame)
        summary_content.pack(fill="both", expand=True)
        
        # Basic info
        info_frame = ttk.Frame(summary_content)
        info_frame.pack(fill="x", pady=5)
        
        rows, cols = self.data_handler.data.shape
        ttk.Label(info_frame, text=f"Rows: {rows}", font=NORMAL_FONT).pack(side="left", padx=20)
        ttk.Label(info_frame, text=f"Columns: {cols}", font=NORMAL_FONT).pack(side="left", padx=20)
        
        # Data types info
        types_frame = ttk.Frame(summary_content)
        types_frame.pack(fill="x", pady=5)
        
        dtypes = self.data_handler.data.dtypes.value_counts()
        dtype_text = "Data Types: " + ", ".join([f"{k} ({v})" for k, v in zip(dtypes.index, dtypes.values)])
        ttk.Label(types_frame, text=dtype_text, font=NORMAL_FONT).pack(side="left", padx=20)
        
        # Missing values info
        missing_frame = ttk.Frame(summary_content)
        missing_frame.pack(fill="x", pady=5)
        
        missing = self.data_handler.data.isna().sum().sum()
        missing_pct = (missing / (rows * cols)) * 100
        ttk.Label(missing_frame, text=f"Missing Values: {missing} ({missing_pct:.2f}%)", font=NORMAL_FONT).pack(side="left", padx=20)
        
        # Column-specific info in a scrollable table
        col_info_frame = ttk.LabelFrame(summary_content, text="Column Information")
        col_info_frame.pack(fill="both", expand=True, pady=10)
        
        col_table = DataTable(col_info_frame)
        col_table.pack(fill="both", expand=True)
        
        # Prepare column info data
        col_info = []
        for col in self.data_handler.get_column_names():
            col_data = self.data_handler.data[col]
            col_type = str(col_data.dtype)
            missing_count = col_data.isna().sum()
            missing_pct = (missing_count / len(col_data)) * 100
            unique_count = col_data.nunique()
            
            # For numerical columns, add statistics
            if pd.api.types.is_numeric_dtype(col_data):
                min_val = col_data.min() if not pd.isna(col_data.min()) else "N/A"
                max_val = col_data.max() if not pd.isna(col_data.max()) else "N/A"
                mean_val = f"{col_data.mean():.2f}" if not pd.isna(col_data.mean()) else "N/A"
                stats = f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}"
            else:
                stats = f"Unique Values: {unique_count}"
            
            col_info.append([col, col_type, f"{missing_count} ({missing_pct:.2f}%)", stats])
        
        # Update the table
        col_table.set_data(
            ["Column Name", "Data Type", "Missing Values", "Statistics"],
            col_info
        )
        
        # Create preprocessing options
        preprocess_content = ttk.Frame(self.preprocess_options_frame)
        preprocess_content.pack(fill="both", expand=True)
        
        # Missing value handling
        missing_label_frame = ttk.LabelFrame(preprocess_content, text="Missing Value Handling")
        missing_label_frame.pack(fill="x", pady=5)
        
        self.missing_var = tk.StringVar(value="mean_median")
        missing_options = [
            ("Mean/Median for numerical, Mode for categorical", "mean_median"),
            ("Zero/Empty string imputation", "zero"),
            ("Drop rows with missing values", "drop")
        ]
        
        for text, value in missing_options:
            ttk.Radiobutton(
                missing_label_frame, 
                text=text, 
                variable=self.missing_var,
                value=value
            ).pack(anchor="w", padx=10, pady=2)
        
        # Categorical encoding
        cat_label_frame = ttk.LabelFrame(preprocess_content, text="Categorical Encoding")
        cat_label_frame.pack(fill="x", pady=5)
        
        self.cat_var = tk.StringVar(value="one_hot")
        cat_options = [
            ("One-Hot Encoding", "one_hot"),
            ("Label Encoding", "label")
        ]
        
        for text, value in cat_options:
            ttk.Radiobutton(
                cat_label_frame, 
                text=text, 
                variable=self.cat_var,
                value=value
            ).pack(anchor="w", padx=10, pady=2)
        
        # Scaling options
        scaling_label_frame = ttk.LabelFrame(preprocess_content, text="Numerical Scaling")
        scaling_label_frame.pack(fill="x", pady=5)
        
        self.scaling_var = tk.StringVar(value="standard")
        scaling_options = [
            ("Standardization (Z-score)", "standard"),
            ("Min-Max Scaling", "minmax"),
            ("No scaling", "none")
        ]
        
        for text, value in scaling_options:
            ttk.Radiobutton(
                scaling_label_frame, 
                text=text, 
                variable=self.scaling_var,
                value=value
            ).pack(anchor="w", padx=10, pady=2)
        
        # Apply preprocessing button
        apply_button = ttk.Button(
            preprocess_content,
            text="Apply Preprocessing",
            command=self.apply_preprocessing,
            style="Accent.TButton"
        )
        apply_button.pack(pady=10)
    
    def _update_eda_tab(self):
        """Update EDA tab with dataset visualizations"""
        # Remove placeholder
        if hasattr(self, 'eda_placeholder') and self.eda_placeholder.winfo_exists():
            self.eda_placeholder.destroy()
        
        # Clear existing content
        for widget in self.eda_content_frame.winfo_children():
            widget.destroy()
        
        # Create EDA content
        # This would typically include visualizations like histograms, bar charts, etc.
        # For now, we'll just add a simple dataset summary
        
        # Data statistics
        stats_frame = ttk.LabelFrame(self.eda_content_frame, text="Dataset Statistics")
        stats_frame.pack(fill="x", pady=10)
        
        # Use pandas describe to get statistics for numerical columns
        try:
            describe_data = self.data_handler.data.describe().transpose()
            describe_table = DataTable(stats_frame)
            describe_table.pack(fill="both", expand=True, pady=10)
            
            # Convert describe data to table format
            headers = ["Column"] + list(describe_data.columns)
            rows = []
            for idx, row in describe_data.iterrows():
                rows.append([idx] + list(row.values))
            
            describe_table.set_data(headers, rows)
        except Exception as e:
            ttk.Label(
                stats_frame,
                text=f"Could not generate statistics: {str(e)}",
                font=NORMAL_FONT
            ).pack(pady=10)
        
        # Correlation analysis (for numerical columns)
        corr_frame = ttk.LabelFrame(self.eda_content_frame, text="Correlation Analysis")
        corr_frame.pack(fill="x", pady=10)
        
        try:
            # Get only numeric columns
            numeric_data = self.data_handler.data.select_dtypes(include=['number'])
            
            if len(numeric_data.columns) > 1:
                corr_data = numeric_data.corr()
                corr_table = DataTable(corr_frame)
                corr_table.pack(fill="both", expand=True, pady=10)
                
                # Convert correlation data to table format
                headers = ["Column"] + list(corr_data.columns)
                rows = []
                for idx, row in corr_data.iterrows():
                    # Format correlation values
                    formatted_row = [f"{val:.2f}" for val in row.values]
                    rows.append([idx] + formatted_row)
                
                corr_table.set_data(headers, rows)
            else:
                ttk.Label(
                    corr_frame,
                    text="Not enough numerical columns for correlation analysis",
                    font=NORMAL_FONT
                ).pack(pady=10)
        except Exception as e:
            ttk.Label(
                corr_frame,
                text=f"Could not generate correlation analysis: {str(e)}",
                font=NORMAL_FONT
            ).pack(pady=10)
        
        # Column value distributions
        dist_frame = ttk.LabelFrame(self.eda_content_frame, text="Column Value Distributions")
        dist_frame.pack(fill="x", pady=10)
        
        # For categorical columns, show value counts
        cat_frame = ttk.Frame(dist_frame)
        cat_frame.pack(fill="x", pady=5)
        
        ttk.Label(cat_frame, text="Categorical Columns:", font=SUBHEADING_FONT).pack(anchor="w", padx=5, pady=5)
        
        # Get categorical columns
        cat_cols = self.data_handler.data.select_dtypes(include=['object', 'category']).columns
        
        if len(cat_cols) > 0:
            for col in cat_cols[:5]:  # Limit to 5 columns for performance
                col_frame = ttk.LabelFrame(cat_frame, text=col)
                col_frame.pack(fill="x", pady=5, padx=10)
                
                # Get value counts
                value_counts = self.data_handler.data[col].value_counts().head(10)  # Top 10 values
                
                # Create a simple text representation
                val_text = "\n".join([f"{val}: {count}" for val, count in zip(value_counts.index, value_counts.values)])
                
                ttk.Label(
                    col_frame,
                    text=val_text,
                    font=NORMAL_FONT,
                    justify=tk.LEFT
                ).pack(anchor="w", padx=10, pady=5)
            
            if len(cat_cols) > 5:
                ttk.Label(
                    cat_frame,
                    text=f"...and {len(cat_cols) - 5} more categorical columns",
                    font=NORMAL_FONT
                ).pack(anchor="w", padx=5, pady=5)
        else:
            ttk.Label(
                cat_frame,
                text="No categorical columns in this dataset",
                font=NORMAL_FONT
            ).pack(anchor="w", padx=5, pady=5)
        
        # For numerical columns, show min, max, mean, etc.
        num_frame = ttk.Frame(dist_frame)
        num_frame.pack(fill="x", pady=5)
        
        ttk.Label(num_frame, text="Numerical Columns:", font=SUBHEADING_FONT).pack(anchor="w", padx=5, pady=5)
        
        # Get numerical columns
        num_cols = self.data_handler.data.select_dtypes(include=['number']).columns
        
        if len(num_cols) > 0:
            for col in num_cols[:5]:  # Limit to 5 columns for performance
                col_frame = ttk.LabelFrame(num_frame, text=col)
                col_frame.pack(fill="x", pady=5, padx=10)
                
                # Get statistics
                col_data = self.data_handler.data[col]
                stats_text = (
                    f"Min: {col_data.min():.2f}\n"
                    f"Max: {col_data.max():.2f}\n"
                    f"Mean: {col_data.mean():.2f}\n"
                    f"Median: {col_data.median():.2f}\n"
                    f"Std: {col_data.std():.2f}"
                )
                
                ttk.Label(
                    col_frame,
                    text=stats_text,
                    font=NORMAL_FONT,
                    justify=tk.LEFT
                ).pack(anchor="w", padx=10, pady=5)
            
            if len(num_cols) > 5:
                ttk.Label(
                    num_frame,
                    text=f"...and {len(num_cols) - 5} more numerical columns",
                    font=NORMAL_FONT
                ).pack(anchor="w", padx=5, pady=5)
        else:
            ttk.Label(
                num_frame,
                text="No numerical columns in this dataset",
                font=NORMAL_FONT
            ).pack(anchor="w", padx=5, pady=5)
    
    def apply_preprocessing(self):
        """Apply preprocessing to the dataset based on selected options"""
        if self.data_handler.data is None:
            messagebox.showerror("Error", "No dataset loaded")
            return
        
        # Start preprocessing animation
        self.status_bar.start_indeterminate("Applying preprocessing...")
        
        # Get preprocessing options
        preprocessing_params = {
            "missing_strategy": self.missing_var.get(),
            "categorical_encoding": self.cat_var.get(),
            "scaling": self.scaling_var.get(),
            "target_column": self.target_column.get()
        }
        
        # Store preprocessing parameters
        self.controller.app_data["preprocessing_params"] = preprocessing_params
        
        # Use a thread to prevent UI freezing
        threading.Thread(
            target=self._preprocess_thread, 
            args=(preprocessing_params,), 
            daemon=True
        ).start()
    
    def _preprocess_thread(self, preprocessing_params):
        """Thread function for preprocessing"""
        try:
            # Apply preprocessing using DataHandler
            self.data_handler.preprocess_data(
                preprocessing_params["missing_strategy"],
                preprocessing_params["categorical_encoding"],
                preprocessing_params["scaling"],
                preprocessing_params["target_column"]
            )
            
            # Update UI on the main thread
            self.after(0, self._update_ui_after_preprocessing)
            
        except Exception as e:
            # Update UI on the main thread with error message
            self.after(0, lambda: self._handle_preprocessing_error(str(e)))
    
    def _handle_preprocessing_error(self, error_message):
        """Handle preprocessing errors"""
        self.status_bar.stop_indeterminate("Error during preprocessing")
        messagebox.showerror("Error", f"Failed to preprocess dataset: {error_message}")
    
    def _update_ui_after_preprocessing(self):
        """Update UI components after preprocessing is complete"""
        # Show a success message
        self.status_bar.stop_indeterminate("Preprocessing complete")
        messagebox.showinfo("Success", "Preprocessing completed successfully")
        
        # Update application data with processed dataset
        self.controller.app_data["processed_dataset"] = self.data_handler.get_processed_data()
        self.controller.app_data["target_column"] = self.target_column.get()
    
    def update_algorithm_list(self):
        """Update the algorithm dropdown based on selected task type"""
        task = self.task_type.get()
        
        if task == "Classification":
            self.algo_combo['values'] = list(CLASSIFICATION_ALGORITHMS.keys())
        else:  # Regression
            self.algo_combo['values'] = list(REGRESSION_ALGORITHMS.keys())
        
        # Set default value
        if self.algo_combo['values']:
            self.algorithm.set(self.algo_combo['values'][0])
    
    def proceed_to_model(self):
        """Proceed to model training page after validation"""
        if self.data_handler.data is None:
            messagebox.showerror("Error", "Please load a dataset first")
            return
        
        if not self.target_column.get():
            messagebox.showerror("Error", "Please select a target column")
            return
        
        if not self.algorithm.get():
            messagebox.showerror("Error", "Please select an algorithm")
            return
        
        # Check if preprocessing has been applied
        if self.controller.app_data.get("processed_dataset") is None:
            result = messagebox.askyesno(
                "Preprocessing Required", 
                "Dataset has not been preprocessed. Apply default preprocessing now?"
            )
            
            if result:
                # Apply default preprocessing
                default_params = {
                    "missing_strategy": "mean_median",
                    "categorical_encoding": "one_hot",
                    "scaling": "standard",
                    "target_column": self.target_column.get()
                }
                
                self.missing_var.set(default_params["missing_strategy"])
                self.cat_var.set(default_params["categorical_encoding"])
                self.scaling_var.set(default_params["scaling"])
                
                self.apply_preprocessing()
                return  # Will be called again after preprocessing
            else:
                return  # User canceled
        
        # Store configuration in application data
        self.controller.app_data["task_type"] = self.task_type.get()
        self.controller.app_data["algorithm"] = self.algorithm.get()
        
        # Proceed to model page
        self.controller.show_frame("model")
    
    def on_tab_changed(self, event):
        """Handle tab change events"""
        selected_tab = self.notebook.index(self.notebook.select())
        
        # Load data for the selected tab if needed
        if selected_tab == 1:  # Preprocessing tab
            if self.data_handler.data is not None and not hasattr(self, 'summary_content'):
                # Trigger preprocessing UI update if it hasn't been done yet
                self._update_preprocessing_tab()
        
        elif selected_tab == 2:  # EDA tab
            if self.data_handler.data is not None and not hasattr(self, 'eda_updated'):
                # Trigger EDA UI update if it hasn't been done yet
                self._update_eda_tab()
                self.eda_updated = True

    def on_show(self):
        """Called when this frame is shown"""
        self.controller.update_status("Configure dataset and preprocessing settings")
