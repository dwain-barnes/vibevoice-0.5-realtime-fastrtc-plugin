# 🎙️ VibeVoice 0.5 Realtime FastRTC Plugin

A FastRTC-compatible wrapper for Microsoft's **VibeVoice-Realtime-0.5B** text-to-speech model, enabling real-time voice streaming in Python applications.

- Model: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B  
- FastRTC: https://fastrtc.org/

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastRTC](https://img.shields.io/badge/FastRTC-compatible-green.svg)](https://fastrtc.org/)
[![Chat](https://github.com/dwain-barnes/vibevoice-0.5-realtime-fastrtc-plugin/blob/main/example_voice_chat.png)

## ✨ Features

- **Real-time streaming** — audio chunks are yielded as they're generated, not after full synthesis
- **FastRTC protocol** — drop-in compatible with FastRTC's `TTSModel` protocol
- **45 voice presets** — English, German, French, and more
- **Low latency** — ~300ms to first audio chunk on GPU
- **Easy integration** — works with FastRTC's `ReplyOnPause`, `Stream`, and WebRTC infrastructure

## 📦 Installation

### Prerequisites

First, install Microsoft VibeVoice:

```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

### Install the plugin

#### From GitHub

```bash
pip install git+https://github.com/dwain-barnes/vibevoice-0.5-realtime-fastrtc-plugin.git
```

#### From source

```bash
git clone https://github.com/dwain-barnes/vibevoice-0.5-realtime-fastrtc-plugin.git
cd vibevoice-0.5-realtime-fastrtc-plugin
pip install -e .
```

#### With FastRTC extras (recommended)

```bash
pip install fastrtc[stt]  # Includes speech-to-text support
```

### Optional dependencies

```bash
pip install sounddevice scipy  # For local audio playback/saving
```

## 🚀 Quick Start

### Basic TTS

```python
from vibevoice_realtime_fastrtc_plugin import VibeVoiceTTS, VibeVoiceOptions

# Initialize (lazy loads model on first use)
tts = VibeVoiceTTS()

# List available voices
print(tts.get_available_voices())

# Generate speech
options = VibeVoiceOptions(speaker_name="en-Emma_woman")
sample_rate, audio = tts.tts("Hello, world!", options)

# Save to file
import scipy.io.wavfile as wav
wav.write("output.wav", sample_rate, audio)
```

### Streaming TTS

```python
from vibevoice_realtime_fastrtc_plugin import VibeVoiceTTS, VibeVoiceOptions

tts = VibeVoiceTTS()
options = VibeVoiceOptions(speaker_name="en-Emma_woman")

# Real-time streaming - chunks yielded as generated
for sample_rate, chunk in tts.stream_tts_sync("This is streaming TTS!", options):
    play_audio(chunk)  # Your audio playback function
```

### FastRTC voice chat

```python
from fastrtc import Stream, ReplyOnPause, get_stt_model
from vibevoice_realtime_fastrtc_plugin import VibeVoiceTTS, VibeVoiceOptions

tts = VibeVoiceTTS()
stt = get_stt_model()
options = VibeVoiceOptions(speaker_name="en-Emma_woman")

def echo(audio):
    text = stt.stt(audio)
    for sample_rate, chunk in tts.stream_tts_sync(text, options):
        audio_array = chunk.reshape(1, -1)
        yield (sample_rate, audio_array, "mono")

stream = Stream(ReplyOnPause(echo), mode="send-receive", modality="audio")
stream.ui.launch()
```

### LLM voice assistant

```python
from fastrtc import Stream, ReplyOnPause, get_stt_model
from vibevoice_realtime_fastrtc_plugin import VibeVoiceTTS, VibeVoiceOptions
import requests

tts = VibeVoiceTTS()
stt = get_stt_model()
options = VibeVoiceOptions(speaker_name="en-Emma_woman")

def get_llm_response(prompt: str) -> str:
    # Works with LM Studio, Ollama, or any OpenAI-compatible API
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def voice_chat(audio):
    text = stt.stt(audio)
    response = get_llm_response(text)
    for sample_rate, chunk in tts.stream_tts_sync(response, options):
        yield (sample_rate, chunk.reshape(1, -1), "mono")

Stream(ReplyOnPause(voice_chat), mode="send-receive", modality="audio").ui.launch()
```

## 🧪 Run examples

```bash
# Echo bot - repeats what you say
python fastrtc_examples.py echo

# LLM voice chat (requires LM Studio or Ollama)
python fastrtc_examples.py chat
```

## 🎛️ Configuration

### `VibeVoiceOptions`

| Parameter      | Type    | Default           | Description |
|---|---|---:|---|
| `speaker_name` | `str`   | `"en-Emma_woman"` | Voice preset name |
| `cfg_scale`    | `float` | `1.5`             | Classifier-Free Guidance scale (`0.0`–`3.0`) |
| `ddpm_steps`   | `int`   | `5`               | Diffusion steps (more = better quality, slower) |

### `VibeVoiceTTS` constructor

| Parameter       | Type    | Default | Description |
|---|---|---:|---|
| `model_path`    | `str`   | `"microsoft/VibeVoice-Realtime-0.5B"` | Hugging Face model ID or local path |
| `device`        | `str`   | `None` (auto)     | `"cuda"`, `"cpu"`, or `"mps"` |
| `output_format` | `str`   | `"float32"`       | `"float32"` or `"int16"` |
| `voices_dir`    | `str`   | `"demo/voices/streaming_model"` | Path to voice preset files |

## 🗣️ Available voices

### English

- `en-Emma_woman` — female, clear and professional
- `en-Carter_man` — male, warm and conversational
- `en-Breeze_woman` — female, soft and gentle
- `en-Mike_man` — male, energetic
- And more…

### Other languages (experimental)

- German: `de-Spk0_man`, `de-Spk1_woman`, etc.
- French: `fr-Spk0_man`, `fr-Spk1_woman`, etc.

List all voices:

```python
from vibevoice_realtime_fastrtc_plugin import VibeVoiceTTS

tts = VibeVoiceTTS()
print(tts.get_available_voices())
```

## 📁 Project structure

```text
vibevoice-0.5-realtime-fastrtc-plugin/
├── examples/
│   └── fastrtc_integration.py
    └── fastrtc_examples.py
├── src/
│   └── vibevoice_realtime_fastrtc_plugin/
│       ├── __init__.py
│       └── vibevoice_tts.py                # Main TTS wrapper
├── LICENSE
├── pyproject.toml
└── README.md
```

## 📊 Performance

| Metric | Value |
|---|---:|
| First chunk latency | ~300ms (GPU) |
| Real-time factor | ~0.3× (faster than real-time) |
| Sample rate | 24kHz |
| Model size | 0.5B parameters |

Tested on an NVIDIA RTX GPU with CUDA.

## 🔧 API Reference

### FastRTC-compatible `TTSModel` surface

```python
from vibevoice_realtime_fastrtc_plugin import VibeVoiceTTS, VibeVoiceOptions

class VibeVoiceTTS:
    def tts(self, text: str, options: VibeVoiceOptions = None) -> tuple[int, "NDArray"]:
        """Generate complete audio from text.

        Returns: (sample_rate, audio_array)
        """

    def stream_tts_sync(self, text: str, options: VibeVoiceOptions = None):
        """Synchronous streaming TTS.

        Yields: (sample_rate, audio_chunk) tuples
        """

    async def stream_tts(self, text: str, options: VibeVoiceOptions = None):
        """Async streaming TTS.

        Yields: (sample_rate, audio_chunk) tuples
        """

    def get_available_voices(self) -> list[str]:
        """List available voice presets."""
```

## 🐛 Troubleshooting

### Flash Attention warning

```text
Flash attention failed... falling back to SDPA
```

This is normal if Flash Attention 2 isn't installed. SDPA works fine with slightly lower performance.

### Tokenizer warning

```text
The tokenizer class you load from this checkpoint is 'Qwen2Tokenizer'...
```

This warning is harmless and does not affect functionality.

### No audio output

- Ensure your audio device is working.
- Check that the FastRTC WebRTC connection is established (ignore "Invalid candidate format" messages).
- Try the echo bot first to test:

```bash
python -m vibevoice_realtime_fastrtc_plugin.examples.fastrtc_integration echo
```

## 📄 License

MIT License — see `LICENSE` for details.

## 🙏 Acknowledgments

- Microsoft VibeVoice — the underlying TTS model
- FastRTC — real-time communication framework
- Hugging Face — model hosting and distribution
