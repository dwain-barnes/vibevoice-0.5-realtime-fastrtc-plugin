"""
VibeVoice-FastRTC
=================

FastRTC-compatible wrapper for Microsoft VibeVoice-Realtime-0.5B TTS.

Quick Start:
    >>> from vibevoice_fastrtc import VibeVoiceTTS, VibeVoiceOptions
    >>> model = VibeVoiceTTS()
    >>> sample_rate, audio = model.tts("Hello, world!")

With FastRTC:
    >>> from fastrtc import Stream, ReplyOnPause
    >>> def handler(audio):
    ...     for chunk in model.stream_tts_sync("Response text"):
    ...         yield chunk
    >>> stream = Stream(ReplyOnPause(handler), mode="send-receive", modality="audio")
"""

from .vibevoice_tts import VibeVoiceTTS, VibeVoiceOptions

__version__ = "0.1.0"
__all__ = [
    "VibeVoiceTTS",
    "VibeVoiceOptions", 

]
