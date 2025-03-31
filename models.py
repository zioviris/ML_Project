from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import mlflow.sklearn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract Base Model for different ML algorithms."""

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

    def evaluate(self, y_true, y_pred):
        """Compute model performance."""
        logger.info("Evaluating model performance.")
        try:
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="binary"),
                "recall": recall_score(y_true, y_pred, average="binary"),
                "f1_score": f1_score(y_true, y_pred, average="binary")
            }
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return None


class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = None  # Model will be set after tuning

    def train(self, X_train, y_train):
        """Train with GridSearchCV to optimize hyperparameters."""
        logger.info("Tuning Random Forest hyperparameters...")
        param_grid = {'n_estimators': [50, 100, 150]}
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logger.info(f"Best params: {grid_search.best_params_}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def log_model_to_mlflow(self, X_train, y_train, model_name):
        with mlflow.start_run():
            logger.info(f"Logging {model_name} to MLflow.")
            self.train(X_train, y_train)
            mlflow.sklearn.log_model(self.model, model_name)


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        """Train with GridSearchCV."""
        logger.info("Tuning Logistic Regression hyperparameters...")
        param_grid = {'C': [0.1, 1, 10]}
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logger.info(f"Best params: {grid_search.best_params_}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def log_model_to_mlflow(self, X_train, y_train, model_name):
        with mlflow.start_run():
            logger.info(f"Logging {model_name} to MLflow.")
            self.train(X_train, y_train)
            mlflow.sklearn.log_model(self.model, model_name)


class SVMModel(BaseModel):
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        """Train with GridSearchCV."""
        logger.info("Tuning SVM hyperparameters...")
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        grid_search = GridSearchCV(SVC(), param_grid, cv=3)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        logger.info(f"Best params: {grid_search.best_params_}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def log_model_to_mlflow(self, X_train, y_train, model_name):
        with mlflow.start_run():
            logger.info(f"Logging {model_name} to MLflow.")
            self.train(X_train, y_train)
            mlflow.sklearn.log_model(self.model, model_name)


# Factory function to dynamically create model instances
def get_model(model_type):
    logger.info(f"Fetching model for type: {model_type}")
    models = {
        "random_forest": RandomForestModel(),
        "logistic_regression": LogisticRegressionModel(),
        "svm": SVMModel()
    }
    return models.get(model_type, None)
