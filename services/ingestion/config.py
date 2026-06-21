from astronova_core.config import get_settings

class IngestionConfig:
    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = "/app/data/uploads"
        
ingestion_config = IngestionConfig()
