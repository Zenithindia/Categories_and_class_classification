import json
from typing import Dict, Any
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"

SYSTEM_RULES = (
    "You are a kids science teacher.\n"
    "Rules:\n"
    "- Use simple English for ages 7-11.\n"
    "- Be factual. If unsure, say 'I’m not sure'.\n"
    "- No scary or graphic details.\n"
    "- Output MUST be valid JSON only. No markdown. No extra text.\n"
)

def explain_animal_ollama(animal: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    prompt = (
        SYSTEM_RULES
        + "\n"
        + f"Animal: {animal}\n"
        + "Context: This animal was predicted from a photo by an animal classifier.\n\n"
        + "Return JSON with keys:\n"
        + "title: string\n"
        + "short: string (1-2 sentences)\n"
        + "facts: array of 5 short bullet strings\n"
        + "habitat: string\n"
        + "food: string\n"
        + "quiz: string (one question for kids)\n"
        + "safety_note: string (1 short sentence about being kind/safe around animals)\n"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "num_predict": 300
        }
    }

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Ollama request failed. Is Ollama running at {OLLAMA_URL}? Error: {e}"
        )

    data = r.json()
    text = (data.get("response") or "").strip()

    # Robust JSON extraction in case model adds extra text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object from the response
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        raise RuntimeError(
            "Model did not return valid JSON. Try a different Ollama model (e.g., qwen2.5:7b) "
            "or reduce temperature."
        )