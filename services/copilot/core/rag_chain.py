import os
import traceback
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


# ─── System prompt for space-weather copilot ───────────────────────────────
SYSTEM_PROMPT = """You are AstroNova Mission Copilot, an expert AI assistant for solar flare forecasting and space weather operations.

You provide authoritative answers about:
- Solar flare classification (A, B, C, M, X classes), GOES X-ray flux levels
- Space weather impacts on Earth (geomagnetic storms, radio blackouts, radiation storms)
- ISRO missions (Aditya-L1, XPoSat, SOLEXS, HEL1OS payloads)
- Satellite operational risk and mitigation strategies
- Solar physics (magnetic reconnection, coronal mass ejections, sunspot cycles)
- NavIC/GPS scintillation and ionospheric effects

Keep responses concise, technical, and actionable. Use proper scientific notation for flux values.
If you don't know something, say so — never fabricate data."""


class SpaceWeatherRAGChain:
    def __init__(self) -> None:
        # Load Gemini configuration from env
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        # Load Ollama configuration from env
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

        # Initialize Gemini client (new google-genai SDK)
        self.genai_client = None
        if self.api_key:
            try:
                from google import genai

                self.genai_client = genai.Client(api_key=self.api_key)
                print(f"[Copilot] Gemini client initialized with model: {self.model_name}")
            except ImportError:
                print("[Copilot] google-genai package not installed. Falling back to Ollama.")
            except Exception as e:
                print(f"[Copilot] Failed to initialize Gemini client: {e}. Falling back to Ollama.")
        else:
            print("[Copilot] No GEMINI_API_KEY found. Using Ollama fallback.")

    def chat(self, query: str) -> dict[str, Any]:
        """Send the user query to Gemini (if configured) or Ollama (fallback) and return a structured response."""

        # ─── Try Gemini first ──────────────────────────────────────────
        if self.genai_client:
            try:
                return self._chat_gemini(query)
            except Exception as e:
                print(f"[Copilot] Gemini call failed: {e}")
                traceback.print_exc()
                # Fall through to Ollama

        # ─── Ollama fallback ───────────────────────────────────────────
        try:
            return self._chat_ollama(query)
        except Exception as e:
            print(f"[Copilot] Ollama call also failed: {e}")
            traceback.print_exc()
            # ─── Final hardcoded fallback ──────────────────────────────
            return self._chat_fallback(query)

    def _chat_gemini(self, query: str) -> dict[str, Any]:
        """Call Google Gemini using the new google-genai SDK."""
        from google.genai import types

        response = self.genai_client.models.generate_content(
            model=self.model_name,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1024,
                temperature=0.1,
            ),
        )
        answer = response.text.strip() if response.text else "No response generated."
        return {"answer": answer, "sources": []}

    def _chat_ollama(self, query: str) -> dict[str, Any]:
        """Call local Ollama LLM as fallback."""
        payload = {
            "model": self.ollama_model,
            "prompt": f"{SYSTEM_PROMPT}\n\nUser: {query}\nAssistant:",
            "stream": False,
        }
        with httpx.Client() as client:
            resp = client.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()
            return {"answer": answer, "sources": []}

    def _chat_fallback(self, query: str) -> dict[str, Any]:
        """Hardcoded intelligent fallback when both Gemini and Ollama are unavailable."""
        q = query.lower()

        if any(kw in q for kw in ["flare", "solar flare", "goes class", "x-ray"]):
            answer = (
                "Solar flares are classified by GOES soft X-ray peak flux:\n"
                "- A-class: < 1e-7 W/m^2 (background)\n"
                "- B-class: 1e-7 to 1e-6 W/m^2\n"
                "- C-class: 1e-6 to 1e-5 W/m^2 (minor)\n"
                "- M-class: 1e-5 to 1e-4 W/m^2 (moderate; can cause HF radio blackouts)\n"
                "- X-class: >= 1e-4 W/m^2 (major; geomagnetic storms and radiation hazards)\n\n"
                "The AstroNova BiLSTM+XGBoost ensemble provides multi-horizon forecasts from 5 min to 24 hr."
            )
        elif any(kw in q for kw in ["cme", "coronal mass", "geomagnetic"]):
            answer = (
                "Coronal Mass Ejections (CMEs) are large-scale expulsions of plasma and magnetic field from the solar corona. "
                "Earth-directed CMEs can trigger geomagnetic storms rated G1 (minor) to G5 (extreme) on the NOAA scale. "
                "Typical transit time to Earth is 1–4 days depending on CME speed (300–3000 km/s). "
                "Impacts include GPS/NavIC scintillation, HF radio blackouts, and satellite drag increases."
            )
        elif any(kw in q for kw in ["aditya", "isro", "solexs", "xposat", "hel1os"]):
            answer = (
                "ISRO's Aditya-L1 is India's first solar observatory, positioned at the Sun-Earth L1 Lagrange point. "
                "Key payloads include:\n"
                "• SOLEXS: Soft X-ray Spectrometer (1–15 keV) for flare monitoring\n"
                "• HEL1OS: Hard X-ray Spectrometer (10–150 keV) for non-thermal emissions\n"
                "• SUIT: Solar Ultraviolet Imaging Telescope\n"
                "• VELC: Visible Emission Line Coronagraph\n\n"
                "XPoSat is India's X-ray polarimetry mission studying cosmic sources."
            )
        elif any(kw in q for kw in ["navic", "gps", "scintillation", "ionosphere"]):
            answer = (
                "Ionospheric scintillation (measured by the S4 index) affects GNSS signals including NavIC and GPS. "
                "During solar flares:\n"
                "• S4 < 0.3: Nominal operations\n"
                "• S4 0.3–0.6: Moderate scintillation; position accuracy degrades\n"
                "• S4 > 0.6: Severe scintillation; potential signal loss\n\n"
                "AstroNova monitors real-time S4 index predictions correlated with X-ray flux forecasts."
            )
        elif any(kw in q for kw in ["hello", "hi", "hey"]):
            answer = (
                "Hello! I'm the AstroNova Mission Copilot. I can help you with:\n"
                "• Solar flare forecasting and GOES classification\n"
                "• Space weather impacts and geomagnetic storm predictions\n"
                "• ISRO Aditya-L1 / XPoSat mission data\n"
                "• Satellite risk assessment and mitigation\n"
                "• NavIC/GPS ionospheric scintillation analysis\n\n"
                "What would you like to know?"
            )
        else:
            answer = (
                f'I received your query: "{query}"\n\n'
                "I'm the AstroNova Space Weather Copilot. I can provide expert analysis on "
                "solar flares, CMEs, geomagnetic storms, ISRO missions (Aditya-L1, SOLEXS), "
                "satellite risk, and ionospheric effects. Please ask me a specific space weather question!"
            )

        return {
            "answer": answer,
            "sources": [{"title": "AstroNova Knowledge Base", "chunk": "Built-in domain knowledge"}],
        }
