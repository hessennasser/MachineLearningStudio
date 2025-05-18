from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression

class ModelTrainer:
    def __init__(self):
        from logic.data_handler import DataHandler
        self.data_handler = DataHandler()  # Initialize DataHandler
        self.model = None
        self.task_type = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def train(self, task="classification", algorithm="KNN"):
        self.task_type = task

        X = self.data_handler.X
        y = self.data_handler.y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if task == "classification":
            if algorithm == "KNN":
                self.model = KNeighborsClassifier()
            elif algorithm == "SVM":
                self.model = SVC()
            elif algorithm == "Decision Tree":
                self.model = DecisionTreeClassifier()
            else:
                raise ValueError("Unsupported classification algorithm.")

        elif task == "regression":
            if algorithm == "Linear Regression":
                self.model = LinearRegression()
            else:
                raise ValueError("Unsupported regression algorithm.")

        else:
            raise ValueError("Task must be either 'classification' or 'regression'.")

        self.model.fit(self.X_train, self.y_train)
