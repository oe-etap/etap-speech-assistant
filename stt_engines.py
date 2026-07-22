from abc import ABC, abstractmethod
import json
import numpy as np

class BaseSTTEngine(ABC):
    @abstractmethod
    def reset(self):
        """
        Reset the engine state (e.g., clear buffers or context) for a new audio stream.
        This should be called before starting latency measurements.
        """
        pass

    @abstractmethod
    def transcribe_stream(self, audio_iterator):
        """
        Accepts an iterator of raw audio byte chunks (16kHz, 16-bit, Mono PCM).
        Yields dictionaries: {"is_final": bool, "text": str}
        """
        pass

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
