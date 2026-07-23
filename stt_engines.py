from abc import ABC, abstractmethod
import json
import numpy as np

class BaseSTTEngine(ABC):
    """Base class for Speech-to-Text engines.

    Subclasses implement a chunk-based streaming interface that accepts raw
    audio bytes and yields transcription results.  Each engine instance is
    used by a single ``stt_worker`` thread at a time — implementations do
    **not** need to be thread-safe, but they must be re-entrant across
    successive ``reset()`` / ``transcribe_stream()`` cycles for different
    audio items.

    To add a new engine (e.g., Conformer), subclass ``BaseSTTEngine``,
    implement ``reset()`` and ``transcribe_stream()``, register the engine
    name in ``mwe_assistant.py``'s argument parser and initialisation block.
    """

    @abstractmethod
    def reset(self):
        """Reset engine state for a new audio stream.

        Called before each audio item to clear any internal buffers,
        decoder state, or accumulated context.  Must be called before
        starting latency measurements.
        """

    @abstractmethod
    def transcribe_stream(self, audio_iterator):
        """Transcribe an audio stream delivered as byte chunks.

        Args:
            audio_iterator: An iterator of ``bytes`` objects, each containing
                raw 16 kHz, 16-bit, mono PCM audio data.

        Yields:
            dict: ``{"is_final": bool, "text": str}``
                - ``is_final=False``: an intermediate (partial) hypothesis
                  that may be revised by later chunks.
                - ``is_final=True``: a stable segment boundary (e.g., after
                  a speech pause detected by VAD).  The text will not change.
        """

class VoskEngine(BaseSTTEngine):
    def __init__(self, model_path):
        from vosk import Model, KaldiRecognizer
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.recognizer.SetWords(True)

    def reset(self):
        self.recognizer.Reset()

    def transcribe_stream(self, audio_iterator):
        for chunk in audio_iterator:
            if self.recognizer.AcceptWaveform(chunk):
                res = json.loads(self.recognizer.Result())
                text = res.get("text", "").strip()
                if text:
                    yield {"is_final": True, "text": text}
            else:
                res = json.loads(self.recognizer.PartialResult())
                text = res.get("partial", "").strip()
                if text:
                    yield {"is_final": False, "text": text}
        
        # End of stream
        res = json.loads(self.recognizer.FinalResult())
        text = res.get("text", "").strip()
        if text:
            yield {"is_final": True, "text": text}

class WhisperEngine(BaseSTTEngine):
    def __init__(self, model_name="small", device="cuda", compute_type="float16"):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def reset(self):
        pass

    def transcribe_stream(self, audio_iterator):
        # Whisper is not natively streaming without complex chunking & VAD.
        # For this refactor, we buffer the entire stream and yield a final result.
        audio_bytes = bytearray()
        for chunk in audio_iterator:
            audio_bytes.extend(chunk)
            # Could yield simulated 'partial' chunks here if we implemented fixed-size windowing.
        
        # Convert bytes to numpy float32 (-1.0 to 1.0)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_data, language="en")
        text = " ".join([s.text.strip() for s in segments]).strip()
        if text:
            yield {"is_final": True, "text": text}
