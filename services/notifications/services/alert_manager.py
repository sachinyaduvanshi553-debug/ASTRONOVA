from typing import Dict, Any, List

class AlertManager:
    def create_alert(self, severity: str, title: str, message: str) -> Dict[str, Any]:
        return {
            "alert_id": "alert_mock_123",
            "severity": severity,
            "title": title,
            "message": message,
            "status": "sent"
        }
