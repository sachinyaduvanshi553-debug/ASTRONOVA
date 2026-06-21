from typing import List, Dict, Any

class SpaceWeatherVectorStore:
    def search(self, query: str) -> List[Dict[str, Any]]:
        # Mock semantic search results
        return [
            {
                "document_id": "doc_1",
                "title": "ISRO Aditya-L1 SoLEXS Instrument Specification",
                "text": "The Solar Low Energy X-ray Spectrometer (SoLEXS) on Aditya-L1 observes soft X-rays in 1-22 keV.",
                "score": 0.89
            },
            {
                "document_id": "doc_2",
                "title": "Space Weather Hazard Guidelines",
                "text": "Solar flares exceeding M5 class pose extreme ionospheric disruption risk for GNSS/NavIC systems.",
                "score": 0.76
            }
        ]
