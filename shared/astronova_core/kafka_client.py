import json
from typing import Any

from confluent_kafka import Producer

from astronova_core.config import get_settings
from astronova_core.logging import get_logger

settings = get_settings()
logger = get_logger("kafka-client")

class AstroNovaProducer:
    def __init__(self):
        conf = {
            'bootstrap.servers': settings.kafka.bootstrap_servers,
            'client.id': 'astronova-producer'
        }
        self.producer = Producer(conf)

    def send_message(self, topic: str, key: str, value: dict[str, Any]) -> None:
        try:
            self.producer.produce(
                topic,
                key=key,
                value=json.dumps(value).encode('utf-8'),
                callback=self._delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            logger.error("failed_to_send_kafka_msg", error=str(e), topic=topic)

    def _delivery_report(self, err, msg):
        if err is not None:
            logger.error("kafka_delivery_failed", error=str(err))
        else:
            logger.debug("kafka_delivery_success", topic=msg.topic(), partition=msg.partition())

    def flush(self):
        self.producer.flush()
