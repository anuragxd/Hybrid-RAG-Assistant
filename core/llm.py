import os
from groq import Groq
from utils.logger import get_logger

logger = get_logger(__name__)

class GroqEngine:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def generate(self, prompt: str, system_prompt: str = "You are an expert research assistant.") -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return ""