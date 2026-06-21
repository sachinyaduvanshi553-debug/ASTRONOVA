from astronova_core.kafka_client import AstroNovaProducer
import json
from typing import Dict, Any

class DataProducer:
    def __init__(self):
        self.producer = AstroNovaProducer()

    def publish_observation(self, key: str, data: Dict[str, Any]) -> None:
        self.producer.send_message("astronova.raw.solexs", key=key, value=data)

    def publish_ingestion_complete(self, job_id: str, stats: Dict[str, Any]) -> None:
        self.producer.send_message("astronova.events", key=job_id, value={
            "event_type": "ingestion_complete",
            "job_id": job_id,
            "stats": stats
        })
