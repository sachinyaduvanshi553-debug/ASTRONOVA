import os
import json
from typing import Any, Dict, List

import google.generativeai as genai
from astronova_core.config import get_settings


class SpaceWeatherRAGChain:
    def __init__(self) -> None:
        settings = get_settings()
        # Load Gemini configuration from settings or env
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment for Copilot service.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _format_sources(self, citations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # Transform Gemini citations into source dicts expected by API consumer
        sources = []
        for c in citations:
            title = c.get("title", "Unknown")
            snippet = c.get("snippet", "")
            sources.append({"title": title, "chunk": snippet})
        return sources

    def chat(self, query: str) -> Dict[str, Any]:
        """Send the user query to Gemini and return a structured response.

        The response includes an ``answer`` string and a list of ``sources`` with ``title`` and ``chunk`` fields.
        """
        # Gemini response may include citations; request them via ``response_metadata``
        try:
            response = self.model.generate_content(
                query,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1024,
                    temperature=0.1,
                ),
                # Enable citation metadata
                request_options={"metadata": {"content_type": "text", "include_citations": True}}
            )
        except Exception as e:
            raise RuntimeError(f"Failed to call Gemini API: {e}")

        answer = response.text.strip()
        # Extract citations if available
        citations = []
        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                if candidate.citation_metadata:
                    for citation in candidate.citation_metadata.citations:
                        citations.append({"title": citation.title, "snippet": citation.snippet})
        sources = self._format_sources(citations)
        return {"answer": answer, "sources": sources}

