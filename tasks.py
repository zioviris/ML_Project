from celery import Celery
from models import get_model
from database import Database
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Celery('tasks', broker='pyamqp://guest@localhost//')

@app.task
def train_model_and_log(model_type, query="SELECT * FROM transactions"):
    logger.info(f"Starting task to train {model_type} model.")
    
    # Fetch data
    db = Database()
    data = db.fetch_data(query)
    if data is None or 'Class' not in data.columns:
        logger.error("Data fetching failed or missing 'Class' column.")
        return

    X = data.drop('Class', axis=1)
    y = data['Class']

    # Train model
    model = get_model(model_type)
    if model is None:
        logger.error(f"Invalid model type: {model_type}")
        return {"error": "Invalid model type"}

    model.log_model_to_mlflow(X, y, model_type)

    logger.info(f"Training completed for {model_type}.")
