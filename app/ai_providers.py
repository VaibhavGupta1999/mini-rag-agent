import os
import json
import requests

class LLMClient:
    def __init__(self):
        # Prefer GROQ if present; fall back to OPENAI-style vars
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.api_key = groq_key or openai_key

        # Default to Groq's OpenAI-compatible endpoint + model
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "llama-3.1-8b-instant")

        # Basic sanity
        self.timeout = 60

    def complete(self, prompt: str):
        """Return a string from the LLM, or None to trigger extractive fallback."""
        if not self.api_key:
            return None
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 700,  # adjust if you want longer answers
            }
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print("LLM error:", e)
            return None
