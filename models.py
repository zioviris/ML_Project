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
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Fraud Detection")


class BaseModel(ABC):
    

    @abstractmethod
    def train(self, X_train: Any, y_train: Any) -> None:
        pass

    @abstractmethod
    def predict(self, X_test: Any) -> Any:
        pass

    def evaluate(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        
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
        client = MlflowClient()  # <-- Added MLflow client instance

        with mlflow.start_run() as run:
            logger.info(f"Training and logging {model_name} to MLflow.")
            self.train(X_train, y_train)
            y_pred = self.predict(X_test)  # <-- predict before log_model for metric calculation
            metrics = self.evaluate(y_test, y_pred)

            if metrics:
                mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(self.best_model, artifact_path="model")  # <-- artifact_path to "model"

            logger.info(f"Logged metrics: {metrics}")

            run_id = run.info.run_id
            model_uri = f"runs:/{run_id}/model"  # <-- "model" to match artifact_path
            try:
                client.create_registered_model(model_name)  # <-- create new registered model
                logger.info(f"Registered new model: {model_name}")
            except RestException:
                logger.info(f"Model {model_name} already registered.")

            registered_model = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)  # <-- Create new model version
            logger.info(f"Registered model {registered_model.name} v{registered_model.version}")

            # Champion/Challenger logic starts here
            try:
                current_prod = client.get_latest_versions(model_name, stages=["Production"])[0]
                current_metrics = client.get_run(current_prod.run_id).data.metrics
                current_f1 = current_metrics.get("f1_score", 0)
                new_f1 = metrics.get("f1_score", 0)

                logger.info(f"Current Production F1: {current_f1} | New F1: {new_f1}")

                if new_f1 > current_f1:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=registered_model.version,
                        stage="Production",
                        archive_existing_versions=True
                    )
                    logger.info("Promoted new model to Production.")
                else:
                    logger.info("ℹNew model NOT promoted. Existing Production model is better.")
            except IndexError:
                # No Production model exists yet
                client.transition_model_version_stage(
                    name=model_name,
                    version=registered_model.version,
                    stage="Production"
                )
                logger.info("First model promoted to Production.")

            return metrics
    
    @staticmethod
    def promote_champion_model(experiment_name: str, model_name: str):
        """Promote best model based on f1_score to Production."""
        logger.info("Selecting champion model based on best F1 score.")
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logger.warning("Experiment not found.")
            return

        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.f1_score DESC"])

        if not runs:
            logger.warning("No runs found.")
            return

        best_run = runs[0]
        best_run_id = best_run.info.run_id

        best_version = None
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.run_id == best_run_id:
                best_version = mv.version
                break

        if best_version:
            logger.info(f"Promoting version {best_version} of model '{model_name}' to Production.")
            client.transition_model_version_stage(
                name=model_name,
                version=best_version,
                stage="Production",
                archive_existing_versions=True
            )
            client.set_model_version_tag(model_name, best_version, "champion", "true")
        else:
            logger.warning("No matching model version found to promote.")


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
