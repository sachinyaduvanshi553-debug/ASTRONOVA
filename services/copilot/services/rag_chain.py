from typing import Any


class SpaceWeatherRAGChain:
    def chat(self, query: str) -> dict[str, Any]:
        # Local RAG-grounded LLM response mock
        return {
            "answer": "Based on the Aditya-L1 SoLEXS Instrument Specifications, the spectrometer monitors solar soft X-ray flux in the 1 to 22 keV range. The anticipated solar flares could cause HF communication blackouts especially over the South-Asian sector.",
            "sources": [
                {"title": "Aditya-L1 SoLEXS specs", "chunk": "observes soft X-rays in 1-22 keV"},
                {"title": "Space Weather Hazard Guidelines", "chunk": "pose extreme ionospheric disruption risk for GNSS/NavIC"}
            ]
        }
