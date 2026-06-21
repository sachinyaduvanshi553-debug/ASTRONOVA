from apscheduler.schedulers.asyncio import AsyncScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from astronova_core.logging import get_logger

logger = get_logger("ingestion-scheduler")

class IngestionScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("scheduler_started")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("scheduler_stopped")
