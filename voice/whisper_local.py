import tempfile
import os
import streamlit as st


@st.cache_resource(show_spinner=False)
def _load_whisper_model():
    """Load Whisper model once and cache it."""
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu", compute_type="int8")


def speech_to_text(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text using faster-whisper.
    Returns transcribed text, or empty string on failure.
    """
    model = _load_whisper_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    try:
        segments, info = model.transcribe(
            path,
            language="en",          # Set to None for auto-detect
            beam_size=3,             # Faster than default beam_size=5
            vad_filter=True,         # Skip silent audio segments (big speed boost)
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")
    finally:
        os.remove(path)
