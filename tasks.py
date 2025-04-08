import eventlet
eventlet.monkey_patch()

from celery import Celery
from models import get_model
from database import Database
import logging
from sklearn.model_selection import train_test_split


# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('tasks', broker='pyamqp://guest@localhost//')

@app.task
def train_model_and_log(model_type, query="SELECT * FROM transactions"):
    logger.info(f"Starting task to train {model_type} model.")
    
    # Fetch data from database
    db = Database()
    data = db.fetch_data(query)
    if data is None or 'Class' not in data.columns:
        logger.error("Data fetching failed or missing 'Class' column.")
        return

    X = data.drop('Class', axis=1)
    y = data['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

    # Get the model based on the model_type
    model = get_model(model_type)
    if model is None:
        logger.error(f"Invalid model type: {model_type}")
        return {"error": "Invalid model type"}

    # Train and log model using MLflow
    try:
        model.log_model_to_mlflow(X_train, y_train, X_test, y_test, model_type) 
        logger.info(f"Training completed for {model_type}. Metrics logged to MLflow.")
    except Exception as e:
        logger.error(f"Error during training or logging for model {model_type}: {e}")
        return {"error": str(e)}

    logger.info(f"Completed task for {model_type}.")

@app.task
def test_mlflow_log():
    import os
    import mlflow

    mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Debug Test")

    with mlflow.start_run() as run:
        mlflow.log_param("dummy_param", 123)
        mlflow.log_metric("dummy_metric", 0.456)
        logger.info(f"Logged dummy MLflow run: {run.info.run_id}")

