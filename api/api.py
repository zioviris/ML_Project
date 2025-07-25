from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

app = FastAPI()

class Features(BaseModel):
    features: List[float]

# --- Automatically fetch the Production model ---
client = MlflowClient()

MODEL_NAME = None
model = None

try:
    # Get all registered models
    registered_models = client.list_registered_models()

    # Find the model that has a version in 'Production' stage
    for rm in registered_models:
        for mv in rm.latest_versions:
            if mv.current_stage == "Production":
                MODEL_NAME = rm.name
                model_uri = f"models:/{MODEL_NAME}/Production"
                model = mlflow.pyfunc.load_model(model_uri=model_uri)
                print(f"Loaded Production model: {MODEL_NAME}")
                break
        if model:  # Exit outer loop once model is found
            break

    if not model:
        print("⚠️ No model in Production stage found.")

except Exception as e:
    print(f"❌ Error loading model from MLflow: {e}")
    model = None


@app.post("/predict")
def predict(data: Features):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(input_arr)[0]

    probabilities = None
    if hasattr(model._model_impl.python_model, "predict_proba"):
        probabilities = model._model_impl.python_model.predict_proba(input_arr)[0].tolist()

    return {
        "prediction": int(prediction),
        "probabilities": probabilities,
        "metadata": {
            "model_name": MODEL_NAME,
            "stage": "Production"
        }
    }
