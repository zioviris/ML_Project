import pandas as pd
from sqlalchemy import create_engine

# Connect to MLflow Postgres DB
engine = create_engine("postgresql://mlflow:mlflow@localhost:5432/mlflow")

# Load runs table
df = pd.read_sql("SELECT * FROM runs", engine)
print(df.head())
