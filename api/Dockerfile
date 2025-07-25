FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY api.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
