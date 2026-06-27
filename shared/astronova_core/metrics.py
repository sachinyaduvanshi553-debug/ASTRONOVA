from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

metrics_router = APIRouter()

REQUEST_COUNT = Counter(
    "astronova_requests_total",
    "Total HTTP Requests",
    ["service", "method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "astronova_request_latency_seconds",
    "HTTP Request Latency",
    ["service", "endpoint"]
)

PREDICTION_COUNT = Counter(
    "astronova_predictions_total",
    "Total Predictions Made",
    ["model_name", "goes_class"]
)

@metrics_router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
