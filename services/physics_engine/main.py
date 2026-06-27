'''Physics Engine Service entry point.
Provides FastAPI application exposing physics-informed calculations.
'''\n\nfrom fastapi import FastAPI\nfrom services.physics_engine.routers import physics\nfrom astronova_core.logging import setup_logging\n\nsetup_logging("physics-engine-service")\n\napp = FastAPI(\n    title="Astronova Physics Engine Service",\n    description="Provides physics-informed metrics for solar flare forecasting.",\n    version="0.1.0"\n)\n\napp.include_router(physics.router)\n\n@app.get("/health")\nasync def health_check():\n    return {"status": "ok"}\n
