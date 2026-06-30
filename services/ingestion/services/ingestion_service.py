import os
import uuid
from datetime import datetime

import pandas as pd
from services.ingestion.models import IngestionJob
from services.ingestion.services.kafka_producer import DataProducer
from sqlalchemy.ext.asyncio import AsyncSession

from astronova_core.logging import get_logger
from astronova_core.models.timeseries import SolexsObservation

logger = get_logger("ingestion-service")

class IngestionService:
    def __init__(self):
        self.producer = DataProducer()

    async def ingest_file(self, file_path: str, file_format: str, db: AsyncSession) -> IngestionJob:
        job = IngestionJob(
            id=uuid.uuid4(),
            status="processing",
            source_file=file_path,
            format=file_format,
            started_at=datetime.utcnow()
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        try:
            # Read file based on format
            if file_format.lower() == "csv":
                df = pd.read_csv(file_path)
            elif file_format.lower() == "json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")

            # Validate columns
            required_cols = ["time", "soft_xray_flux", "hard_xray_flux"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            df["time"] = pd.to_datetime(df["time"])

            # Store in DB and publish to Kafka
            rows_stored = 0
            for _, row in df.iterrows():
                obs_time = row["time"].to_pydatetime()
                # Store raw observation
                db_obs = SolexsObservation(
                    time=obs_time,
                    soft_xray_flux=float(row["soft_xray_flux"]),
                    hard_xray_flux=float(row["hard_xray_flux"]),
                    energy_band_lo=1.0,
                    energy_band_hi=8.0,
                    quality_flag=int(row.get("quality_flag", 0)),
                    source_file=os.path.basename(file_path)
                )
                await db.merge(db_obs)

                # Publish raw data to Kafka topic for Processing Service
                self.producer.publish_observation(
                    key=obs_time.isoformat(),
                    data={
                        "time": obs_time.isoformat(),
                        "soft_xray_flux": float(row["soft_xray_flux"]),
                        "hard_xray_flux": float(row["hard_xray_flux"]),
                        "quality_flag": int(row.get("quality_flag", 0)),
                        "source_file": os.path.basename(file_path)
                    }
                )
                rows_stored += 1

            # Update job status
            job.status = "completed"
            job.rows_ingested = rows_stored
            job.completed_at = datetime.utcnow()
            await db.commit()

            # Publish ingestion complete event
            self.producer.publish_ingestion_complete(str(job.id), {"rows_ingested": rows_stored})
            logger.info("ingestion_job_success", job_id=str(job.id), rows=rows_stored)

        except Exception as e:
            logger.error("ingestion_job_failed", job_id=str(job.id), error=str(e))
            job.status = "failed"
            job.errors = [str(e)]
            job.completed_at = datetime.utcnow()
            await db.commit()

        return job
