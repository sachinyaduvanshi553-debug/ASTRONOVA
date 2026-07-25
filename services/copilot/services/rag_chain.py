import os
import json
import httpx
from typing import Any, Dict, List

from astronova_core.config import get_settings
from dotenv import load_dotenv

load_dotenv()


class SpaceWeatherRAGChain:
    def __init__(self) -> None:
        self.settings = get_settings()
        # Load Gemini configuration from settings or env
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        # Load Ollama configuration
        ollama_settings = self.settings.ollama()
        self.ollama_url = getattr(ollama_settings, "base_url", "http://localhost:11434")
        self.ollama_model = getattr(ollama_settings, "model", "llama3.2:3b")

        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            # We don't raise an error anymore, instead we will fallback to Ollama
            self.model = None

    def _format_sources(self, citations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # Transform Gemini citations into source dicts expected by API consumer
        sources = []
        for c in citations:
            title = c.get("title", "Unknown")
            snippet = c.get("snippet", "")
            sources.append({"title": title, "chunk": snippet})
        return sources

    def chat(self, query: str) -> Dict[str, Any]:
        """Send the user query to Gemini (if configured) or Ollama (fallback) and return a structured response."""
        if self.model:
            try:
                import google.generativeai as genai
                response = self.model.generate_content(
                    query,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=1024,
                        temperature=0.1,
                    ),
                    request_options={"metadata": {"content_type": "text", "include_citations": True}}
                )
                answer = response.text.strip()
                citations = []
                if hasattr(response, "candidates") and response.candidates:
                    for candidate in response.candidates:
                        if candidate.citation_metadata:
                            for citation in candidate.citation_metadata.citations:
                                citations.append({"title": citation.title, "snippet": citation.snippet})
                sources = self._format_sources(citations)
                return {"answer": answer, "sources": sources}
            except Exception as e:
                return {"answer": f"Error calling Gemini API: {e}", "sources": []}
        else:
            # Fallback to Ollama
            try:
                payload = {
                    "model": self.ollama_model,
                    "prompt": query,
                    "stream": False
                }
                with httpx.Client() as client:
                    resp = client.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30.0)
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("response", "")
                    return {"answer": answer.strip(), "sources": []}
            except Exception as e:
                return {"answer": f"Error calling Ollama API (no Gemini key found): {e}", "sources": []}

