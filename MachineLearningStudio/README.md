# Machine Learning Studio

A Python-based desktop application built with Tkinter that allows users to upload datasets (CSV files), configure machine learning tasks (classification or regression), train models, and evaluate their performance—all within a user-friendly GUI.

## Features

### Welcome Screen
- Project title display
- Start button to navigate to the configuration page
- Modern UI with light and dark themes

### Dataset Upload & Configuration
- CSV file upload functionality
- Automatic column detection and preview
- Task type selection (Classification/Regression)
- Dynamic algorithm selection based on task type
- Preprocessing options for handling missing values
- Automated feature encoding and normalization

### Model Training & Evaluation
- Train models with selected algorithms
- Performance evaluation with appropriate metrics:
  - Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
  - Regression: MAE, RMSE, R² Score, Predicted vs Actual plot
- Save and load trained models
- Manual prediction interface for testing

## Installation

1. Clone this repository:
```
git clone https://github.com/hessennasser/MachineLearningStudio.git
cd MachineLearningStudio
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
python main.py
```

## Requirements
- Python 3.6+
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- Pillow
- Tkinter (usually comes with Python)

## Usage

1. Start the application
2. Upload a CSV dataset
3. Select the task type and target column
4. Configure preprocessing options
5. Train the model
6. Evaluate performance and make predictions

## Project Structure

- `assets/`: Contains icons, images, and styles
- `config/`: Configuration settings
- `core/`: Data handling, model training, and evaluation
- `gui/`: Tkinter UI components and pages
- `main.py`: Application entry point