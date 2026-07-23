#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM engine abstraction layer.
Provides OllamaEngine for streaming text generation via the Ollama API.
"""

import json
import requests


# Stop sequences to prevent the model from generating synthetic multi-turn dialogue
_DEFAULT_STOP_TOKENS = ["\nUser:", "\nHuman:", "\n---", "---", "<|end|>", "<|user|>"]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a concise, factual but friendly voice assistant. "
    "Answer the user's question in English in 1-3 medium length sentences. "
    "Do not simulate a conversation. Do not generate follow-up questions "
    "or responses from the user. Provide only your single answer."
)


class OllamaEngine:
    """Ollama-based LLM engine with streaming generation and cancellation support."""

    def __init__(self, model: str, url: str, system_prompt: str = None,
                 stop_tokens: list = None, max_tokens: int = 150,
                 temperature: float = 0.7):
        """Initialize the Ollama engine.

        Args:
            model: Ollama model name (e.g. "phi3:mini").
            url: Ollama generate endpoint URL.
            system_prompt: Custom system prompt (uses default if None).
            stop_tokens: Stop sequences to prevent multi-turn hallucination.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
        """
        self._model = model
        self._url = url
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._stop_tokens = stop_tokens if stop_tokens is not None else _DEFAULT_STOP_TOKENS
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._session = requests.Session()
        self._current_response = None

    def warmup(self):
        """Send a minimal request to load the model into VRAM/RAM."""
        try:
            self._session.post(self._url, json={
                "model": self._model,
                "prompt": "",
                "options": {"num_predict": 1}
            }, timeout=120)
        except Exception as e:
            print(f"[WARN] Failed to warmup Ollama: {e}")

    def cancel(self):
        """Cancel the current streaming response by closing the HTTP connection."""
        resp = self._current_response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
            self._current_response = None

    def generate_stream(self, user_text: str):
        """Stream LLM response, yielding sentence-level chunks.

        Yields dicts with keys:
            - "text" (str): The synthesizable text chunk.
            - "ollama_stats" (dict|None): Server-side metrics (only on last yield).
            - "cancelled" (bool): True if the stream was cancelled mid-generation.

        The last yielded dict may have empty "text" and non-None "ollama_stats"
        containing Ollama's server-side performance metrics.
        """
        # Cancel any previous stream before starting a new one
        self.cancel()

        prompt = (
            f'{self._system_prompt}\n\n'
            f'The user said: "{user_text}"\n\n'
            f'Answer:'
        )

        try:
            r = self._session.post(self._url, json={
                "model": self._model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict": self._max_tokens,
                    "temperature": self._temperature,
                    "stop": self._stop_tokens,
                }
            }, timeout=120)
            r.raise_for_status()
            self._current_response = r

            buffer = ""
            terminators = {".", "?", "!", ":", ";", "\n"}
            ollama_stats = None

            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    piece = data.get("response", "")
                    buffer += piece

                    # Capture server-side stats from the final message
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
                            yield {"text": buffer.strip(), "ollama_stats": None,
                                   "cancelled": False}
                            buffer = ""

            # Flush remaining buffer
            if buffer.strip():
                yield {"text": buffer.strip(), "ollama_stats": None, "cancelled": False}

            # Yield final metadata-only entry with server-side stats
            if ollama_stats:
                yield {"text": "", "ollama_stats": ollama_stats, "cancelled": False}

        except requests.exceptions.ChunkedEncodingError:
            # Expected when cancel() closes the response mid-stream
            yield {"text": "", "ollama_stats": None, "cancelled": True}
        except Exception as e:
            yield {"text": f"(LLM call failed: {e})", "ollama_stats": None,
                   "cancelled": False}
        finally:
            self._current_response = None
