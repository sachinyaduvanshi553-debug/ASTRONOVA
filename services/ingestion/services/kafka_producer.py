from typing import Any

from astronova_core.kafka_client import AstroNovaProducer


class DataProducer:
    def __init__(self):
        self.producer = AstroNovaProducer()

    def publish_observation(self, key: str, data: dict[str, Any]) -> None:
        self.producer.send_message("astronova.raw.solexs", key=key, value=data)

    def publish_ingestion_complete(self, job_id: str, stats: dict[str, Any]) -> None:
        self.producer.send_message("astronova.events", key=job_id, value={
            "event_type": "ingestion_complete",
            "job_id": job_id,
            "stats": stats
        })
