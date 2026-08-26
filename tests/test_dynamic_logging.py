import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.abspath("."))

from astronova_core.dynamic_logging import dynamic_logger_manager

from services.gateway.main import app

client = TestClient(app)


def test_dynamic_logging_manager_levels():
    """Test dynamic log level getters and setters."""
    original_level = dynamic_logger_manager.get_level("gateway")
    assert original_level in ["INFO", "DEBUG", "WARNING", "ERROR"]

    # Set log level to DEBUG
    updated = dynamic_logger_manager.set_level("gateway", "DEBUG")
    assert updated == "DEBUG"
    assert dynamic_logger_manager.get_level("gateway") == "DEBUG"

    # Reset back to INFO
    dynamic_logger_manager.set_level("gateway", "INFO")
    assert dynamic_logger_manager.get_level("gateway") == "INFO"


def test_dynamic_logging_buffer_and_push():
    """Test pushing dynamic logs and querying recent logs."""
    dynamic_logger_manager.clear_logs()

    entry = dynamic_logger_manager.push_log(
        service_name="test_service",
        level="WARNING",
        category="UNIT_TEST",
        message="Test dynamic log message",
        method="GET",
        path="/test",
        status_code=200,
    )

    assert entry["service_name"] == "test_service"
    assert entry["level"] == "WARNING"

    recent = dynamic_logger_manager.get_recent_logs(service_name="test_service")
    assert len(recent) >= 1
    assert recent[0]["message"] == "Test dynamic log message"


def test_api_get_log_levels():
    """Test GET /api/v1/logging/level API endpoint."""
    response = client.get("/api/v1/logging/level")
    assert response.status_code == 200
    data = response.json()
    assert "gateway" in data or "global" in data


def test_api_set_log_level():
    """Test POST /api/v1/logging/level API endpoint."""
    payload = {"service_name": "gateway", "level": "DEBUG"}
    response = client.post("/api/v1/logging/level", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["service_name"] == "gateway"
    assert data["current_level"] == "DEBUG"


def test_api_post_custom_log_and_query():
    """Test creating a log via POST and retrieving it via GET /api/v1/logging/logs."""
    payload = {
        "service_name": "postman_service",
        "level": "INFO",
        "category": "API_TEST",
        "message": "Postman simulated log",
        "method": "POST",
        "path": "/api/v1/test",
        "status_code": 201,
    }
    post_res = client.post("/api/v1/logging/logs", json=payload)
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "success"

    get_res = client.get("/api/v1/logging/logs?service_name=postman_service")
    assert get_res.status_code == 200
    logs_data = get_res.json()
    assert logs_data["count"] >= 1
    assert logs_data["logs"][0]["message"] == "Postman simulated log"


def test_api_get_db_schema():
    """Test GET /api/v1/logging/db/schema endpoint for Postman inspection."""
    response = client.get("/api/v1/logging/db/schema")
    assert response.status_code == 200
    data = response.json()
    assert "database_driver" in data
    assert "tables" in data
    assert len(data["tables"]) > 0


def test_api_execute_dynamic_query():
    """Test POST /api/v1/logging/db/query endpoint for read-only SELECT execution."""
    payload = {"query": "SELECT id, service_name, level, message FROM dynamic_logs", "limit": 10}
    response = client.post("/api/v1/logging/db/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "columns" in data
    assert "rows" in data
    assert "execution_time_ms" in data


def test_api_db_query_reject_non_select():
    """Ensure non-SELECT statements (e.g. DROP, DELETE, INSERT) are rejected for security."""
    payload = {"query": "DELETE FROM dynamic_logs", "limit": 10}
    response = client.post("/api/v1/logging/db/query", json=payload)
    assert response.status_code == 400
    assert "Only read-only queries" in response.json()["detail"]


def test_api_db_stats():
    """Test GET /api/v1/logging/db/stats performance & health endpoint."""
    response = client.get("/api/v1/logging/db/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "total_buffered_logs" in data
