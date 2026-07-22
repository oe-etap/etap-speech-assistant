import requests
import json

class OllamaEngine:
    def __init__(self, model_name="phi3:mini", url="http://localhost:11434/api/generate"):
        self.model_name = model_name
        self.url = url
        self.session = requests.Session()
        self.warmup()

    def warmup(self):
        try:
            self.session.post(self.url, json={
                "model": self.model_name,
                "prompt": "",
                "options": {"num_predict": 1}
            }, timeout=120)
        except Exception as e:
            print(f"[WARN] Failed to warmup Ollama: {e}")

    def generate_stream(self, text_prompt):
        """
        Streams LLM responses. 
        Yields dicts:
          - 'text': the synthesizable text chunk (str)
          - 'ollama_stats': None for text chunks; dict with server-side timing for the final metadata chunk
        """
        system_prompt = (
            "You are a concise, factual but friendly voice assistant. "
            "Answer in English in 1-3 medium length sentences."
        )
        
        try:
            r = self.session.post(self.url, json={
                "model": self.model_name,
                "system": system_prompt,
                "prompt": text_prompt, 
                "stream": True, 
                "options": {
                    "num_predict": 150,
                    "temperature": 0.7
                }
            }, timeout=120)
            r.raise_for_status()
            
            buffer = ""
            terminators = {".", "?", "!", ":", ";", "\n"}
            ollama_stats = None
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    piece = data.get("response", "")
                    buffer += piece
                    
                    if data.get("done", False):
                        ollama_stats = {
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
                            "eval_count": data.get("eval_count", 0),
                            "eval_duration_ns": data.get("eval_duration", 0),
                            "total_duration_ns": data.get("total_duration", 0),
                        }
                    
                    if any(t in piece for t in terminators):
                        if len(buffer.strip()) > 2:
                            yield {"text": buffer.strip(), "ollama_stats": None}
                            buffer = ""
            if buffer.strip():
                yield {"text": buffer.strip(), "ollama_stats": None}
            if ollama_stats:
                yield {"text": "", "ollama_stats": ollama_stats}
        except Exception as e:
            yield {"text": f"(LLM call failed: {e})", "ollama_stats": None}
