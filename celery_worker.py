import eventlet
eventlet.monkey_patch()

from celery import Celery


celery_app = Celery(
    "tasks",
    broker= "amqp://guest:guest@localhost:5672//",
    backend="rpc://",
)

@celery_app.task
def add(x, y):
    return x + y

# Directly reference the Celery task
# no need to wrap it in a separate task here
celery_app.autodiscover_tasks(["tasks"])

