"""
Data handling and preprocessing module for ML Analyzer
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataHandler:
    """
    Handles dataset loading, preprocessing, and transformation
    """
    def __init__(self):
        """Initialize the DataHandler"""
        self.data = None
        self.original_data = None
        self.preprocessed_data = None
        self.preprocessing_pipeline = None
        self.target_column = None
        self.feature_columns = None
        self.feature_names = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.datetime_columns = None
    
    def load_dataset(self, file_path):
        """
        Load a dataset from a CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try to infer dtypes for more efficient loading
            self.data = pd.read_csv(file_path)
            self.original_data = self.data.copy()
            
            # Reset any previous preprocessing
            self.preprocessed_data = None
            self.preprocessing_pipeline = None
            
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def get_preview(self, rows=10):
        """
        Get a preview of the dataset
        
        Args:
            rows: Number of rows to return
            
        Returns:
            DataFrame: Preview of the dataset
        """
        if self.data is None:
            return None
        
        return self.data.head(rows)
    
    def get_column_names(self):
        """
        Get the column names of the dataset
        
        Returns:
            list: Column names
        """
        if self.data is None:
            return []
        
        return self.data.columns.tolist()
    
    def identify_column_types(self):
        """
        Identify the types of columns in the dataset
        
        Returns:
            tuple: (categorical_columns, numerical_columns, datetime_columns)
        """
        if self.data is None:
            return [], [], []
        
        # Identify numerical columns (exclude possible datetime columns)
        self.numerical_columns = self.data.select_dtypes(include=['int', 'float']).columns.tolist()
        
        # Identify categorical columns
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Try to identify datetime columns
        self.datetime_columns = []
        for col in self.data.columns:
            # Check column name hints
            if any(date_str in col.lower() for date_str in ['date', 'time', 'year', 'month', 'day']):
                # Try to convert to datetime
                try:
                    pd.to_datetime(self.data[col])
                    self.datetime_columns.append(col)
                    # Remove from other types if it was added
                    if col in self.numerical_columns:
                        self.numerical_columns.remove(col)
                    if col in self.categorical_columns:
                        self.categorical_columns.remove(col)
                except:
                    pass
        
        return self.categorical_columns, self.numerical_columns, self.datetime_columns
    
    def preprocess_data(self, missing_strategy='mean_median', categorical_encoding='one_hot', 
                        scaling='standard', target_column=None):
        """
        Preprocess the dataset for machine learning
        
        Args:
            missing_strategy: Strategy for handling missing values
            categorical_encoding: Strategy for encoding categorical variables
            scaling: Strategy for scaling numerical features
            target_column: Name of the target column
            
        Returns:
            tuple: (X, y) preprocessed features and target
        """
        if self.data is None:
            raise ValueError("No dataset loaded")
        
        if target_column is None:
            raise ValueError("Target column must be specified")
        
        self.target_column = target_column
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Store original feature columns
        self.feature_columns = X.columns.tolist()
        
        # Identify column types if not already done
        if self.categorical_columns is None or self.numerical_columns is None:
            self.identify_column_types()
        
        # Filter out datetime columns and target column from feature lists
        cat_cols = [col for col in self.categorical_columns if col != target_column and col not in self.datetime_columns]
        num_cols = [col for col in self.numerical_columns if col != target_column and col not in self.datetime_columns]
        
        # Create preprocessing steps
        preprocessor_steps = []
        
        # Numerical features preprocessing
        if num_cols:
            if missing_strategy == 'mean_median':
                num_imputer = SimpleImputer(strategy='mean')
            elif missing_strategy == 'zero':
                num_imputer = SimpleImputer(strategy='constant', fill_value=0)
            elif missing_strategy == 'drop':
                # Handle in a separate step
                X = X.dropna()
                y = y.loc[X.index]
                num_imputer = SimpleImputer(strategy='mean')  # Fallback
            else:
                num_imputer = SimpleImputer(strategy='mean')
            
            if scaling == 'standard':
                num_scaler = StandardScaler()
            elif scaling == 'minmax':
                num_scaler = MinMaxScaler()
            else:  # 'none'
                num_scaler = None
            
            if num_scaler:
                num_pipeline = Pipeline([
                    ('imputer', num_imputer),
                    ('scaler', num_scaler)
                ])
            else:
                num_pipeline = Pipeline([
                    ('imputer', num_imputer)
                ])
            
            preprocessor_steps.append(('num', num_pipeline, num_cols))
        
        # Categorical features preprocessing
        if cat_cols:
            if missing_strategy == 'mean_median' or missing_strategy == 'drop':
                cat_imputer = SimpleImputer(strategy='most_frequent')
            elif missing_strategy == 'zero':
                cat_imputer = SimpleImputer(strategy='constant', fill_value='')
            else:
                cat_imputer = SimpleImputer(strategy='most_frequent')
            
            if categorical_encoding == 'one_hot':
                cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            elif categorical_encoding == 'label':
                cat_encoder = LabelEncoder()
                # Label encoding is handled differently because it works on 1D
                for col in cat_cols:
                    X[col] = X[col].fillna('unknown')
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                cat_cols = []  # Clear as they're already processed
            else:
                cat_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            
            if cat_cols:  # Only if we have categorical columns left to process
                if categorical_encoding == 'one_hot':
                    cat_pipeline = Pipeline([
                        ('imputer', cat_imputer),
                        ('encoder', cat_encoder)
                    ])
                    preprocessor_steps.append(('cat', cat_pipeline, cat_cols))
        
        # Create and apply the full preprocessing pipeline
        if preprocessor_steps:
            self.preprocessing_pipeline = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='drop'  # Drop any columns not explicitly included
            )
            
            # Fit and transform
            X_transformed = self.preprocessing_pipeline.fit_transform(X)
            
            # Get feature names
            self.feature_names = self._get_feature_names(self.preprocessing_pipeline, X.columns)
        else:
            # If no preprocessing needed, just convert to array
            X_transformed = X.values
            self.feature_names = X.columns.tolist()
        
        # Store preprocessed data
        self.preprocessed_data = (X_transformed, y.values)
        
        return self.preprocessed_data
    
    def _get_feature_names(self, column_transformer, original_features):
        """
        Get feature names from sklearn ColumnTransformer
        
        Args:
            column_transformer: fitted ColumnTransformer
            original_features: original feature names
            
        Returns:
            list: transformed feature names
        """
        feature_names = []
        
        for name, pipe, cols in column_transformer.transformers_:
            if name == 'remainder':
                # Skip remainder
                continue
                
            if hasattr(pipe, 'get_feature_names_out'):
                # For newer scikit-learn versions
                if name == 'cat' and isinstance(pipe.named_steps.get('encoder'), OneHotEncoder):
                    # For one-hot encoding
                    transformed_names = pipe.get_feature_names_out(cols)
                    feature_names.extend(transformed_names)
                else:
                    # For other transformers
                    feature_names.extend(pipe.get_feature_names_out(cols))
            elif hasattr(pipe, 'get_feature_names'):
                # For older scikit-learn versions
                if name == 'cat' and isinstance(pipe.named_steps.get('encoder'), OneHotEncoder):
                    # For one-hot encoding
                    transformed_names = pipe.named_steps['encoder'].get_feature_names(cols)
                    feature_names.extend(transformed_names)
                else:
                    # For other transformers
                    feature_names.extend(cols)
            else:
                # Default: use original column names
                feature_names.extend(cols)
        
        # If no feature names were extracted, fall back to index-based names
        if not feature_names:
            feature_names = [f"feature_{i}" for i in range(column_transformer.transform(pd.DataFrame(
                0, index=[0], columns=original_features)).shape[1])]
        
        return feature_names
    
    def get_processed_data(self):
        """
        Get the preprocessed data
        
        Returns:
            tuple: (X, y) preprocessed features and target
        """
        if self.preprocessed_data is None:
            raise ValueError("Data has not been preprocessed yet")
        
        return self.preprocessed_data
    
    def transform_new_data(self, new_data):
        """
        Apply preprocessing to new data
        
        Args:
            new_data: DataFrame with new data to transform
            
        Returns:
            array: Transformed data
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessing pipeline not created yet")
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in new_data.columns:
                new_data[col] = np.nan
        
        # Apply transformation
        return self.preprocessing_pipeline.transform(new_data[self.feature_columns])
    
    def get_data_summary(self):
        """
        Get a summary of the dataset
        
        Returns:
            dict: Summary statistics
        """
        if self.data is None:
            return None
        
        summary = {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "missing_values": self.data.isna().sum().sum(),
            "dtypes": self.data.dtypes.value_counts().to_dict(),
            "column_stats": {}
        }
        
        # Add per-column statistics
        for col in self.data.columns:
            col_data = self.data[col]
            col_summary = {
                "dtype": str(col_data.dtype),
                "missing": col_data.isna().sum(),
                "unique_values": col_data.nunique()
            }
            
            # Add numerical statistics if appropriate
            if pd.api.types.is_numeric_dtype(col_data):
                col_summary.update({
                    "min": col_data.min() if not pd.isna(col_data.min()) else None,
                    "max": col_data.max() if not pd.isna(col_data.max()) else None,
                    "mean": col_data.mean() if not pd.isna(col_data.mean()) else None,
                    "median": col_data.median() if not pd.isna(col_data.median()) else None,
                    "std": col_data.std() if not pd.isna(col_data.std()) else None
                })
            
            summary["column_stats"][col] = col_summary
        
        return summary
