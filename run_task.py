# run_task.py
from tasks import train_model_and_log

# Trigger the task (use an existing model_type like "random_forest")
train_model_and_log.delay("logistic_regression")
