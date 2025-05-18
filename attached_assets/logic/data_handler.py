import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataHandler:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.encoders = {}
        self.scaler = None

    def set_dataframe(self, df: pd.DataFrame):
        self.df = df.copy()

    def preprocess(self):
        if self.df is None:
            raise ValueError("No data loaded.")

        # handle missing values
        self.df.fillna(self.df.mode().iloc[0], inplace=True)

        # sleect numeric and categorical columns
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = self.df.select_dtypes(include=["object"]).columns

        # encode categorical columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders[col] = le

        # scale numeric columns
        self.scaler = StandardScaler()
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])

    def get_processed_data(self):
        return self.df

    def set_target_column(self, target_col):
        if self.df is None:
            raise ValueError("No data loaded.")

        self.y = self.df[target_col]
        self.X = self.df.drop(columns=[target_col])
