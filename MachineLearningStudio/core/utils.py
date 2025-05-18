"""
Utility functions for the ML Analyzer application
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime
import json
import csv

def validate_dataset(data):
    """
    Validate that a dataset is suitable for machine learning
    
    Args:
        data: Pandas DataFrame to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if data is None:
        return False, "No data provided"
    
    if len(data) == 0:
        return False, "Dataset is empty"
    
    if len(data.columns) < 2:
        return False, "Dataset needs at least two columns (one feature and one target)"
    
    # Check for excessive missing values
    missing_percentages = data.isna().mean() * 100
    for col, pct in missing_percentages.items():
        if pct > 50:
            return False, f"Column '{col}' has more than 50% missing values"
    
    return True, ""

def guess_task_type(target_series):
    """
    Try to guess whether a problem is classification or regression
    
    Args:
        target_series: Series containing target values
        
    Returns:
        str: 'Classification' or 'Regression'
    """
    # Remove missing values for analysis
    target_series = target_series.dropna()
    
    # Count unique values
    n_unique = target_series.nunique()
    
    # If only a few unique values or object dtype, likely classification
    if n_unique <= 10 or pd.api.types.is_object_dtype(target_series):
        return 'Classification'
    elif pd.api.types.is_numeric_dtype(target_series):
        # If numeric with many unique values, likely regression
        if n_unique > len(target_series) * 0.1:  # More than 10% are unique
            return 'Regression'
        else:
            # Could be either, but more likely classification
            return 'Classification'
    else:
        # Default to classification for other types
        return 'Classification'

def get_optimal_encoding(column, threshold=10):
    """
    Determine the best encoding for a categorical column
    
    Args:
        column: Pandas Series to analyze
        threshold: Maximum number of unique values for one-hot encoding
        
    Returns:
        str: 'one_hot' or 'label'
    """
    n_unique = column.nunique()
    
    if n_unique <= threshold:
        return 'one_hot'
    else:
        return 'label'

def format_number(value):
    """
    Format a number for display (handles None and NaN)
    
    Args:
        value: Number to format
        
    Returns:
        str: Formatted number
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    elif isinstance(value, (float, np.floating)):
        if abs(value) < 0.001 or abs(value) >= 10000:
            return f"{value:.4e}"
        else:
            return f"{value:.4f}"
    else:
        return str(value)

def export_to_csv(data, file_path):
    """
    Export data to a CSV file
    
    Args:
        data: Data to export (DataFrame, dict, or list)
        file_path: Path to save the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame.from_dict(data, orient='index').reset_index()
            df.columns = ['Metric', 'Value']
            df.to_csv(file_path, index=False)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # List of dicts
            pd.DataFrame(data).to_csv(file_path, index=False)
        else:
            # Fallback for other data types
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                if isinstance(data, list):
                    for row in data:
                        writer.writerow(row if isinstance(row, (list, tuple)) else [row])
                else:
                    writer.writerow([data])
        
        return True
    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False

def generate_timestamp():
    """
    Generate a formatted timestamp
    
    Returns:
        str: Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(directory):
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        directory: Directory path
        
    Returns:
        bool: True if directory exists or was created
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return True
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    return True

def replace_values_in_svg(svg_string, replacements):
    """
    Replace placeholder values in an SVG string
    
    Args:
        svg_string: Original SVG string
        replacements: Dictionary of {placeholder: replacement} pairs
        
    Returns:
        str: Updated SVG string
    """
    result = svg_string
    
    for placeholder, replacement in replacements.items():
        result = result.replace(placeholder, replacement)
    
    return result

def create_heatmap_figure(data, title="Correlation Heatmap", figsize=(8, 6)):
    """
    Create a heatmap figure for correlation matrix
    
    Args:
        data: 2D numpy array or pandas DataFrame
        title: Title for the heatmap
        figsize: Figure size as (width, height) tuple
        
    Returns:
        Figure: Matplotlib figure object
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Create heatmap
    im = ax.imshow(data, cmap='coolwarm')
    
    # Add colorbar
    cbar = fig.colorbar(im)
    
    # Add labels
    ax.set_title(title)
    
    # Add ticks
    if isinstance(data, pd.DataFrame):
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.columns)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.columns)
    else:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
    
    # Rotate x axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    if data.shape[0] <= 10 and data.shape[1] <= 10:  # Only add text for smaller matrices
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data.iloc[i, j]:.2f}" if isinstance(data, pd.DataFrame) 
                        else f"{data[i, j]:.2f}",
                        ha="center", va="center", color="black" if abs(data.iloc[i, j] if isinstance(data, pd.DataFrame) 
                                                                        else data[i, j]) < 0.5 else "white")
    
    fig.tight_layout()
    return fig

def is_valid_filename(filename):
    """
    Check if a filename is valid
    
    Args:
        filename: Filename to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check for invalid characters
    if re.search(r'[<>:"/\\|?*]', filename):
        return False
    
    # Check if filename is empty or just dots
    if not filename or filename.strip('.') == '':
        return False
    
    return True
