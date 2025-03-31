FROM python:3.9-slim

# Install flower
RUN pip install flower

# Set the working directory for the container
WORKDIR /app

# Install other necessary dependencies if required
# RUN pip install -r requirements.txt

# Start flower with the proper broker configuration
CMD ["flower", "--broker=redis://redis:6379/0", "--port=5555"]

