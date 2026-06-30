from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from astronova_core.database import Base


class SolexsObservation(Base):
    __tablename__ = "solexs_observations"

    time = Column(DateTime, primary_key=True)
    soft_xray_flux = Column(Float, nullable=False)
    hard_xray_flux = Column(Float, nullable=False)
    energy_band_lo = Column(Float, nullable=False)
    energy_band_hi = Column(Float, nullable=False)
    quality_flag = Column(Integer, default=0)
    source_file = Column(String(255), nullable=True)
    data_version = Column(String(50), default="1.0.0")

class ProcessedObservation(Base):
    __tablename__ = "processed_observations"

    time = Column(DateTime, primary_key=True)
    cleaned_soft_flux = Column(Float, nullable=False)
    cleaned_hard_flux = Column(Float, nullable=False)
    interpolated = Column(Boolean, default=False)
    outlier_removed = Column(Boolean, default=False)
    processing_pipeline_id = Column(String(100), nullable=False)
