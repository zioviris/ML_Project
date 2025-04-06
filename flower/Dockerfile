FROM python:3.9-slim

# Install flower
RUN pip install flower

# Set the working directory
WORKDIR /app

# Start flower with RabbitMQ broker
CMD ["flower", "--broker=amqp://guest:guest@rabbitmq:5672//", "--port=5555"]

