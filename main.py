from fastapi import FastAPI, BackgroundTasks
import sqlite3
import pandas as pd
from celery_worker import add
from celery.result import AsyncResult
from pydantic import BaseModel
from tasks import train_model_and_log 
from tasks import test_mlflow_log

app = FastAPI()

DATABASE_PATH = "data/creditcard_fraud.db"

def query_db(query: str):
    """Helper function to execute SQL queries and return results as DataFrame."""
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Card Fraud API"}

@app.get("/transactions")
def get_all_transactions():
    df = query_db("SELECT * FROM transactions LIMIT 100") 
    return df.to_dict(orient="records")

@app.get("/transaction/{transaction_id}")
def get_transaction(transaction_id: int):
    df = query_db(f"SELECT * FROM transactions WHERE rowid = {transaction_id}")
    if df.empty:
        return {"error": "Transaction not found"}
    return df.to_dict(orient="records")[0]

@app.get("/fraud-transactions")
def get_fraud_transactions():
    df = query_db("SELECT * FROM transactions WHERE Class = 1 LIMIT 100")
    return df.to_dict(orient="records")

class TrainRequest(BaseModel):
    model_type: str  # Expecting 'random_forest', 'logistic_regression', or 'svm'

@app.post("/train_model")
async def train_model(model_type: str):
    """Trigger model training in Celery."""
    if model_type not in ["random_forest", "logistic_regression", "svm"]:
        return {"error": "Invalid model type"}
    
    # Start the task asynchronously in Celery
    task = train_model_and_log.apply_async(args=[model_type])
    
    # Return the task ID and the model type to track the training
    return {"task_id": task.id, "message": f"Training started for {model_type}"}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """Check the status of a Celery task and fetch model metrics if completed."""
    result = AsyncResult(task_id)

    # If the task is complete, fetch the results (including metrics)
    if result.status == "SUCCESS":
        task_result = result.result
        if isinstance(task_result, dict) and 'metrics' in task_result:
            return {
                "task_id": task_id,
                "status": result.status,
                "metrics": task_result["metrics"]
            }
        else:
            return {
                "task_id": task_id,
                "status": result.status,
                "message": "Training completed without logging metrics."
            }
    else:
        return {"task_id": task_id, "status": result.status}

@app.get("/test-mlflow")
def test_mlflow_endpoint():
    task = test_mlflow_log.delay()
    return {"task_id": task.id, "message": "Dummy MLflow log task submitted"}
