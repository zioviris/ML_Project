# Fraud Detection — Minikube Kubernetes Deployment (ephemeral)

## Quickstart (PowerShell)
```powershell
minikube start
& minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t backend:latest -f Dockerfile .
docker build -t celery-worker:latest -f Dockerfile.celery .
docker build -t mlflow:latest -f mlflow/Dockerfile .
kubectl apply -f k8s/ --recursive
kubectl get pods -n fraud-detection -w
```
