"""
RoadSense Alert Service
Manages audio warning generation and visual alert dispatching.
Uses dual audio architecture:
1. In-memory PCM WAV synthesis played via native Streamlit st.audio(..., autoplay=True)
   for guaranteed, loud speaker alarms across all browsers.
2. Web Speech API browser speech synthesis for clear spoken collision warnings.
"""
import io
import wave
import math
import struct
from dataclasses import dataclass
from typing import Optional, Dict, Any
from config import AlertConfig
from utils.logging_config import logger


@dataclass
class AlertEvent:
    """Dispatched alert event metadata."""
    timestamp: float
    risk_level: str
    track_id: Optional[int]
    message: str
    audio_played: bool


class AlertService:
    """
    Coordinates audio alarms, synthesized chimes, speech warnings, and rate-limiting.
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.is_muted: bool = not self.config.audio_enabled

    def toggle_mute(self) -> bool:
        """Toggles audio mute status. Returns new muted state."""
        self.is_muted = not self.is_muted
        logger.info(f"Audio mute state toggled to: {self.is_muted}")
        return self.is_muted

    def set_muted(self, muted: bool) -> None:
        self.is_muted = muted

    def generate_wav_beep(self, frequencies: list[float], duration_sec: float = 0.25, sample_rate: int = 44100) -> bytes:
        """
        Synthesizes an in-memory WAV audio chime with smooth envelope attack/decay.
        """
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)        # Mono
            wf.setsampwidth(2)        # 16-bit PCM
            wf.setframerate(sample_rate)

            total_samples = int(sample_rate * duration_sec)
            seg_len = total_samples // len(frequencies)

            for seg_idx, freq in enumerate(frequencies):
                for i in range(seg_len):
                    t = float(i) / sample_rate
                    # Envelope window to avoid clicks
                    envelope = 1.0
                    attack = int(seg_len * 0.15)
                    decay = int(seg_len * 0.25)
                    if i < attack:
                        envelope = float(i) / attack
                    elif i > seg_len - decay:
                        envelope = float(seg_len - i) / decay

                    val = int(math.sin(2.0 * math.pi * freq * t) * 32767.0 * 0.8 * envelope)
                    val = max(-32767, min(32767, val))
                    wf.writeframes(struct.pack('<h', val))

        return buf.getvalue()

    def get_alert_wav(self, risk_level: str) -> Optional[bytes]:
        """
        Returns high-priority acoustic warning sounds for different risk states.
        """
        if self.is_muted:
            return None

        if "HIGH" in risk_level:
            # Urgent alternating dual siren pulse (880 Hz -> 1180 Hz -> 880 Hz -> 1180 Hz)
            return self.generate_wav_beep([950.0, 1300.0, 950.0, 1300.0], duration_sec=0.35)
        elif "MEDIUM" in risk_level:
            # Gentle two-tone caution chime (587 Hz -> 784 Hz)
            return self.generate_wav_beep([587.0, 784.0], duration_sec=0.28)
        else:
            # Information test chime
            return self.generate_wav_beep([523.0, 659.0, 784.0], duration_sec=0.30)

    def get_test_sound_wav(self) -> bytes:
        """Generates a pleasant 3-tone acoustic test chime (C5 -> E5 -> G5)."""
        return self.generate_wav_beep([523.25, 659.25, 783.99, 1046.50], duration_sec=0.45)

    def generate_browser_speech_html(self, message: str, risk_level: str) -> str:
        """
        Generates an embedded HTML script for browser Web Speech synthesis.
        """
        if self.is_muted:
            return ""

        speech_text = message.replace('"', '\\"').replace("'", "\\'")
        pitch = 1.2 if "HIGH" in risk_level else 1.0

        return f"""
        <div style="display:none;" id="roadsense-speech-{int(id(message) % 100000)}">
            <script>
            (function() {{
                try {{
                    if ('speechSynthesis' in window) {{
                        window.speechSynthesis.cancel();
                        const utterance = new SpeechSynthesisUtterance("{speech_text}");
                        utterance.rate = 1.15;
                        utterance.pitch = {pitch};
                        utterance.volume = 1.0;
                        window.speechSynthesis.speak(utterance);
                    }}
                }} catch (e) {{
                    console.warn("Speech synthesis prevented:", e);
                }}
            }})();
            </script>
        </div>
        """
