from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn
import logging
import os
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variable for MLflow tracking URI
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Fraud Detection")


class BaseModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> None:
        pass

    @abstractmethod
    def predict(self, X_test: Any) -> Any:
        pass

    def evaluate(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        """Evaluate model performance using classification metrics."""
        logger.info("Evaluating model performance.")
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0)
            }
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {}


class ModelWithGridSearch(BaseModel):
    """Model with hyperparameter tuning using GridSearchCV."""

    def __init__(self, model: Any, param_grid: Dict[str, Any]):
        self.model = model
        self.param_grid = param_grid
        self.best_model = None

    def train(self, X_train: Any, y_train: Any) -> None:
        logger.info("Tuning hyperparameters using GridSearchCV...")
        grid_search = GridSearchCV(self.model, self.param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")

    def predict(self, X_test: Any) -> Any:
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_model.predict(X_test)

    def log_model_to_mlflow(self, X_train: Any, y_train: Any, X_test: Any, y_test: Any, model_name: str) -> Dict[str, float]:
        """Logs model and metrics to MLflow and returns metrics."""
        with mlflow.start_run():
            logger.info(f"Training and logging {model_name} to MLflow.")
            self.train(X_train, y_train)
            mlflow.sklearn.log_model(self.best_model, model_name)

            y_pred = self.predict(X_test)
            metrics = self.evaluate(y_test, y_pred)

            if metrics:
                mlflow.log_metrics(metrics)

            logger.info(f"Logged metrics: {metrics}")
            return metrics


class RandomForestModel(ModelWithGridSearch):
    def __init__(self):
        param_grid = {'n_estimators': [50, 100, 150]}
        super().__init__(RandomForestClassifier(), param_grid)


class LogisticRegressionModel(ModelWithGridSearch):
    def __init__(self):
        param_grid = {'C': [0.1, 1, 10]}
        super().__init__(LogisticRegression(max_iter=1000), param_grid)


class SVMModel(ModelWithGridSearch):
    def __init__(self):
        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        super().__init__(SVC(), param_grid)


def get_model(model_type: str) -> Optional[BaseModel]:
    logger.info(f"Fetching model for type: {model_type}")
    models = {
        "random_forest": RandomForestModel(),
        "logistic_regression": LogisticRegressionModel(),
        "svm": SVMModel()
    }
    return models.get(model_type)
