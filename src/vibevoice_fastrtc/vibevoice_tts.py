"""
VibeVoice-Realtime-0.5B FastRTC Wrapper (Robust & Thread-Safe)
==============================================================

FastRTC-compatible wrapper for Microsoft VibeVoice-Realtime-0.5B.
Includes built-in text sanitization, thread locking, and interruption handling.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class VibeVoiceOptions:
    """Configuration options for VibeVoice TTS."""
    speaker_name: str = "en-Emma_woman"
    cfg_scale: float = 1.5
    ddpm_steps: int = 5


class StopGenerationException(Exception):
    """Custom exception to cleanly halt model generation."""
    pass


class VibeVoiceTTS:
    """FastRTC-compatible wrapper for VibeVoice-Realtime-0.5B."""
    
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        model_path: str = "microsoft/VibeVoice-Realtime-0.5B",
        device: Optional[str] = None,
        output_format: str = "float32",
        voices_dir: Optional[str] = None,
    ):
        self.model_path = model_path
        self.output_format = output_format
        self.voices_dir = voices_dir or "demo/voices/streaming_model"
        
        self._model = None
        self._processor = None
        self._voice_cache = {}
        
        # Global lock to prevent the model from crashing if called from multiple threads
        self.lock = threading.Lock()
        
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"VibeVoiceTTS initialized (device={self.device})")
    
    def _sanitize_text(self, text: str) -> str:
        """
        Internal cleaner to prevent model crashes.
        """
        if not text:
            return ""
            
        replacements = {
            "’": "'", "‘": "'", "“": '"', "”": '"', 
            "…": "...", "—": ", ", "–": ", ", "-": ", " 
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        text = re.sub(r'\([^\)]*\)', '', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        text = text.replace("*", "")
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!\']', '', text)
        text = " ".join(text.split())
        
        return text

    def _reset_model_state(self):
        """Reset model internal state to prevent index errors after interruption."""
        if self._model is not None:
            try:
                import torch
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Clear any internal hooks or temp states
                self._model.zero_grad(set_to_none=True)
            except Exception as e:
                logger.warning(f"Failed to reset model state: {e}")

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is not None:
            return
        
        import torch
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor
        )
        
        logger.info(f"Loading VibeVoice from {self.model_path}...")
        
        if self.device == "mps":
            dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            dtype = torch.float32
            attn_impl = "sdpa"
        
        try:
            def load_with_attn(impl):
                return VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map="cuda" if self.device == "cuda" else None,
                    attn_implementation=impl,
                )

            self._model = load_with_attn(attn_impl)
            if self.device == "mps": self._model.to("mps")
            elif self.device == "cpu": self._model.to("cpu")
                
        except Exception as e:
            if attn_impl == "flash_attention_2":
                logger.warning(f"Flash attention failed, falling back to SDPA. Error: {e}")
                self._model = load_with_attn("sdpa")
                if self.device == "mps": self._model.to("mps")
            else:
                raise
        
        self._model.eval()
        self._processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)
        logger.info("Model loaded successfully")
    
    def _get_voice(self, speaker_name: str) -> dict:
        if speaker_name in self._voice_cache:
            return self._voice_cache[speaker_name]
        
        import torch
        speaker_lower = speaker_name.lower()
        voices_path = Path(self.voices_dir)
        if not voices_path.exists():
            voices_path = Path("demo/voices/streaming_model")
        
        voice_path = None
        exact_path = voices_path / f"{speaker_lower}.pt"
        
        if exact_path.exists():
            voice_path = exact_path
        else:
            for pt_file in voices_path.glob("*.pt"):
                if speaker_lower in pt_file.stem.lower() or pt_file.stem.lower() in speaker_lower:
                    voice_path = pt_file
                    break
        
        if voice_path is None:
            default_path = voices_path / "en-emma_woman.pt"
            if default_path.exists():
                voice_path = default_path
            else:
                 raise FileNotFoundError(f"Voice '{speaker_name}' not found.")
        
        cached_prompt = torch.load(voice_path, map_location=self.device, weights_only=False)
        self._voice_cache[speaker_name] = cached_prompt
        return cached_prompt
    
    def _format_audio(self, audio) -> NDArray:
        import torch
        if isinstance(audio, torch.Tensor):
            audio = audio.float().cpu().numpy()
        
        if audio is None or len(audio) == 0:
            return np.array([], dtype=np.float32)
        
        audio = np.squeeze(audio)
        if audio.ndim > 1: audio = audio.flatten()
        
        if self.output_format == "int16":
            if audio.dtype in (np.float32, np.float64):
                audio = np.clip(audio, -1.0, 1.0)
                return (audio * 32767).astype(np.int16)
            return audio.astype(np.int16)
        else:
            if audio.dtype == np.int16:
                return audio.astype(np.float32) / 32767.0
            return audio.astype(np.float32)
    
    def get_available_voices(self) -> list[str]:
        voices_path = Path(self.voices_dir)
        if not voices_path.exists():
            voices_path = Path("demo/voices/streaming_model")
        if voices_path.exists():
            return sorted([f.stem for f in voices_path.glob("*.pt")])
        return []

    # ------------------------------------------------------------------------
    # ASYNC WRAPPER
    # ------------------------------------------------------------------------
    async def stream_tts(self, text: str, options: Optional[VibeVoiceOptions] = None) -> AsyncGenerator[tuple[int, NDArray], None]:
        loop = asyncio.get_event_loop()
        gen = self.stream_tts_sync(text, options)
        try:
            while True:
                # Use run_in_executor to avoid blocking the async event loop with next()
                chunk = await loop.run_in_executor(None, next, gen)
                yield chunk
        except StopIteration:
            pass
        except Exception:
            # If the async task is cancelled, we ensure the sync generator gets closed
            gen.close()
            raise

    # ------------------------------------------------------------------------
    # SYNC STREAMING (The Logic Core)
    # ------------------------------------------------------------------------
    def stream_tts_sync(self, text: str, options: Optional[VibeVoiceOptions] = None) -> Generator[tuple[int, NDArray], None, None]:
        """Synchronous streaming TTS with Robust Thread Safety."""
        text = self._sanitize_text(text)
        if not text.strip():
            return

        self._load_model()
        options = options or VibeVoiceOptions()

        import copy
        import torch
        from vibevoice.modular.streamer import AudioStreamer

        # 1. ACQUIRE LOCK
        # Ensure no other generation is running.
        self.lock.acquire()

        # Flags for thread control
        stop_event = threading.Event()
        generation_done = threading.Event()
        audio_queue = queue.Queue()
        
        # We define errors here to re-raise them after lock release if needed
        generation_error = None

        try:
            self._model.set_ddpm_inference_steps(num_steps=options.ddpm_steps)
            all_prefilled_outputs = self._get_voice(options.speaker_name)

            inputs = self._processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=all_prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)

            # --- Custom Streamer that supports stopping ---
            class InterruptibleStreamer(AudioStreamer):
                def __init__(self, q, format_fn, stop_evt):
                    super().__init__(batch_size=1)
                    self.q = q
                    self.format_fn = format_fn
                    self.stop_evt = stop_evt

                def put(self, audio_chunk, *args, **kwargs):
                    # CHECK FOR STOP SIGNAL
                    if self.stop_evt.is_set():
                        raise StopGenerationException("Generation interrupted by user.")

                    if audio_chunk is not None:
                        if isinstance(audio_chunk, torch.Tensor):
                            audio_chunk = audio_chunk.float().cpu().numpy()
                        formatted = self.format_fn(audio_chunk)
                        if len(formatted) > 0:
                            self.q.put(formatted)

                def end(self, *args, **kwargs):
                    pass

            streamer = InterruptibleStreamer(audio_queue, self._format_audio, stop_event)

            # --- Generation Thread ---
            def generate():
                nonlocal generation_error
                try:
                    with torch.no_grad():
                        self._model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=options.cfg_scale,
                            tokenizer=self._processor.tokenizer,
                            generation_config={'do_sample': False},
                            verbose=False,
                            all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs),
                            audio_streamer=streamer,
                        )
                except StopGenerationException:
                    # This is a clean exit, do not log error
                    pass
                except (IndexError, RuntimeError) as e:
                    # Only log if we didn't ask it to stop
                    if not stop_event.is_set():
                        generation_error = e
                        logger.error(f"Generation interrupted/error: {e}")
                except Exception as e:
                    generation_error = e
                    logger.error(f"Unexpected generation error: {e}")
                finally:
                    generation_done.set()

            gen_thread = threading.Thread(target=generate, daemon=True)
            gen_thread.start()

            # --- Yield Loop ---
            while not generation_done.is_set() or not audio_queue.empty():
                try:
                    # Check for stop signal from the consumer side
                    if stop_event.is_set():
                        break

                    chunk = audio_queue.get(timeout=0.05)
                    yield self.SAMPLE_RATE, chunk
                except queue.Empty:
                    continue

        except GeneratorExit:
            # Handle case where consumer (FastRTC) closes the generator
            stop_event.set()
        except Exception as e:
            stop_event.set()
            logger.error(f"Stream setup error: {e}")
        finally:
            # 2. CLEANUP & RELEASE LOCK
            # Signal thread to stop
            stop_event.set()
            
            # CRITICAL: Wait for the generation thread to actually finish/die.
            # If we release the lock before this thread dies, the next request 
            # will crash into this one (IndexError).
            if 'gen_thread' in locals() and gen_thread.is_alive():
                gen_thread.join(timeout=2.0) 

            # Force model reset if we stopped early
            self._reset_model_state()

            if self.lock.locked():
                self.lock.release()