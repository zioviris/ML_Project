# config.py

# SQLite database path
DATABASE_PATH = 'data/creditcard_fraud.db'

# MLflow configuration
MLFLOW_TRACKING_URI = 'http://localhost:5000'  # Default MLflow UI URI

# Use RabbitMQ instead of Redis
CELERY_BROKER_URL = "amqp://guest:guest@rabbitmq:5672//"

