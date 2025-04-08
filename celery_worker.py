import eventlet
eventlet.monkey_patch()

from celery import Celery


celery_app = Celery(
    "tasks",
    broker="pyamqp://guest@localhost//",
    backend="rpc://",
)

@celery_app.task
def add(x, y):
    return x + y

# Directly reference the Celery task
# no need to wrap it in a separate task here
celery_app.autodiscover_tasks(["tasks"])

