FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host=0.0.0.0", \
     "--port=5000", \
     "--backend-store-uri=postgresql://mlflow:mlflow@postgres:5432/mlflow", \
     "--default-artifact-root=/app/mlruns"]
