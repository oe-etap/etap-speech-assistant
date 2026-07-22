from abc import ABC, abstractmethod
import subprocess
import os
import json
import numpy as np

class BaseTTSEngine(ABC):
    @abstractmethod
    def synthesize_stream(self, text_iterator):
        """
        Accepts an iterator of text chunks.
        Yields tuples: (pcm_bytes, sample_rate)
        """
        pass

class PiperEngine(BaseTTSEngine):
    def __init__(self, piper_exe, piper_voice, use_exe=False):
        self.piper_exe = piper_exe
        self.piper_voice = piper_voice
        self.use_exe = use_exe
        
        json_path = self.piper_voice + ".json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Piper configuration JSON not found: {json_path}. This file is mandatory.")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            vdata = json.load(f)
            self.sample_rate = vdata["audio"]["sample_rate"]

        self.piper_voice_obj = None
        if not self.use_exe:
            from piper.voice import PiperVoice
            self.piper_voice_obj = PiperVoice.load(self.piper_voice)

    def synthesize_stream(self, text_iterator):
        for text in text_iterator:
            text = text.strip()
            if not text:
                continue
            
            if self.use_exe:
                cmd = [self.piper_exe, "-m", self.piper_voice, "--output_raw"]
                result = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.PIPE, check=True)
                yield result.stdout, self.sample_rate
            else:
                pcm_parts = []
                for audio_chunk in self.piper_voice_obj.synthesize(text):
                    pcm_parts.append(audio_chunk.audio_int16_bytes)
                yield b"".join(pcm_parts), self.sample_rate

# Constant for converting float32 audio [-1.0, 1.0] to 16-bit PCM (signed 16-bit max value)
INT16_MAX = 32767.0

class CoquiEngine(BaseTTSEngine):
    def __init__(self, voice="xtts_v2", language="en", speaker="Daisy Studious"):
        from TTS.api import TTS
        self.tts = TTS(model_name=f"tts_models/multilingual/multi-dataset/{voice}")
        self.language = language
        self.speaker = speaker

    def synthesize_stream(self, text_iterator):
        for text in text_iterator:
            text = text.strip()
            if not text:
                continue
            
            wav = self.tts.tts(text=text, speaker=self.speaker, language=self.language)
            audio_data = np.array(wav, dtype=np.float32)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * INT16_MAX).astype(np.int16)
            sample_rate = self.tts.synthesizer.output_sample_rate
            yield audio_data.tobytes(), sample_rate
