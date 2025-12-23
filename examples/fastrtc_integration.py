#!/usr/bin/env python3
"""
FastRTC + VibeVoice Integration Examples
=========================================

This file demonstrates various ways to use VibeVoice-Realtime-0.5B
with the FastRTC real-time communication framework.

Usage:
    python examples/fastrtc_integration.py --example echo
    python examples/fastrtc_integration.py --example llm
    python examples/fastrtc_integration.py --example websocket
"""

import argparse
import logging
from pathlib import Path

# Add parent directory to path for local development
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vibevoice_fastrtc import VibeVoiceTTS, VibeVoiceOptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_echo_bot():
    """Simple echo bot - repeats back what you say."""
    from fastrtc import Stream, ReplyOnPause, get_stt_model
    
    logger.info("Initializing Echo Bot...")
    
    # Initialize models
    tts = VibeVoiceTTS()
    stt = get_stt_model()
    options = VibeVoiceOptions(
        speaker_name="Emma",
        cfg_scale=1.5,
        ddpm_steps=5,
    )
    
    def echo_handler(audio):
        """Transcribe and speak back."""
        text = stt.stt(audio)
        logger.info(f"User said: {text}")
        
        for chunk in tts.stream_tts_sync(text, options):
            yield chunk
    
    # Create stream with Gradio UI
    stream = Stream(
        ReplyOnPause(echo_handler),
        mode="send-receive",
        modality="audio",
    )
    
    logger.info("Launching Echo Bot...")
    stream.ui.launch()


def example_llm_chat():
    """Voice chat with an LLM (using Anthropic Claude)."""
    from fastrtc import Stream, ReplyOnPause, get_stt_model
    
    try:
        import anthropic
    except ImportError:
        print("Install anthropic: pip install anthropic")
        return
    
    logger.info("Initializing LLM Voice Chat...")
    
    # Initialize
    tts = VibeVoiceTTS()
    stt = get_stt_model()
    client = anthropic.Anthropic()
    
    options = VibeVoiceOptions(
        speaker_name="Mike",
        cfg_scale=1.5,
    )
    
    # Conversation history for multi-turn
    conversation = []
    
    def chat_handler(audio):
        """Process voice input and get LLM response."""
        # Transcribe
        user_text = stt.stt(audio)
        logger.info(f"User: {user_text}")
        
        # Add to conversation
        conversation.append({"role": "user", "content": user_text})
        
        # Get LLM response
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system="You are a helpful voice assistant. Keep responses concise and conversational.",
            messages=conversation,
        )
        
        assistant_text = response.content[0].text
        conversation.append({"role": "assistant", "content": assistant_text})
        logger.info(f"Assistant: {assistant_text}")
        
        # Stream TTS response
        for chunk in tts.stream_tts_sync(assistant_text, options):
            yield chunk
    
    stream = Stream(
        ReplyOnPause(chat_handler),
        mode="send-receive",
        modality="audio",
    )
    
    logger.info("Launching LLM Voice Chat...")
    stream.ui.launch()


def example_streaming_text():
    """Demo of streaming text input (simulating LLM token streaming)."""
    import time
    import numpy as np
    
    logger.info("Streaming Text Demo...")
    
    tts = VibeVoiceTTS()
    options = VibeVoiceOptions(speaker_name="Grace")
    
    # Simulate streaming text from an LLM
    text = "Hello! This is a demonstration of streaming text to speech. "
    text += "Each word arrives one at a time, just like tokens from a language model."
    
    words = text.split()
    
    logger.info("Simulating streaming LLM output...")
    for i, word in enumerate(words):
        partial_text = " ".join(words[:i+1])
        print(f"\rText: {partial_text}", end="", flush=True)
        time.sleep(0.1)  # Simulate LLM token delay
    
    print("\n\nGenerating audio...")
    
    # Generate complete audio
    sample_rate, audio = tts.tts(text, options)
    
    # Save to file
    import scipy.io.wavfile as wav
    output_path = "streaming_demo.wav"
    wav.write(output_path, sample_rate, audio)
    logger.info(f"Saved audio to {output_path}")
    logger.info(f"Duration: {len(audio)/sample_rate:.2f}s")


def example_multi_speaker():
    """Demo of different voice presets."""
    import scipy.io.wavfile as wav
    from pathlib import Path
    
    logger.info("Multi-Speaker Demo...")
    
    tts = VibeVoiceTTS()
    
    # Test different speakers
    speakers = ["Carter", "Emma", "Mike", "Grace"]
    text = "Hello, my name is {}. I'm a voice preset for VibeVoice."
    
    output_dir = Path("speaker_samples")
    output_dir.mkdir(exist_ok=True)
    
    for speaker in speakers:
        logger.info(f"Generating sample for {speaker}...")
        options = VibeVoiceOptions(speaker_name=speaker)
        
        sample_rate, audio = tts.tts(text.format(speaker), options)
        
        output_path = output_dir / f"{speaker.lower()}_sample.wav"
        wav.write(str(output_path), sample_rate, audio)
        logger.info(f"  Saved: {output_path}")
    
    logger.info(f"All samples saved to {output_dir}/")


def example_websocket_server():
    """Run a WebSocket server for real-time TTS."""
    import asyncio
    import json
    
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets")
        return
    
    logger.info("Starting WebSocket TTS Server...")
    
    tts = VibeVoiceTTS(output_format="int16")
    
    async def handler(websocket):
        logger.info("Client connected")
        
        async for message in websocket:
            try:
                data = json.loads(message)
                text = data.get("text", "")
                speaker = data.get("speaker", "Emma")
                
                logger.info(f"Generating TTS for: {text[:50]}...")
                
                options = VibeVoiceOptions(speaker_name=speaker)
                
                # Stream audio chunks
                async for sample_rate, chunk in tts.stream_tts(text, options):
                    # Send as binary (int16 PCM)
                    await websocket.send(chunk.tobytes())
                
                # Send completion signal
                await websocket.send(json.dumps({"status": "complete"}))
                
            except Exception as e:
                logger.error(f"Error: {e}")
                await websocket.send(json.dumps({"error": str(e)}))
    
    async def main():
        async with websockets.serve(handler, "localhost", 8765):
            logger.info("WebSocket server running on ws://localhost:8765")
            logger.info("Send JSON: {\"text\": \"Hello world\", \"speaker\": \"Emma\"}")
            await asyncio.Future()  # Run forever
    
    asyncio.run(main())


def example_benchmark():
    """Benchmark TTS performance."""
    import time
    
    logger.info("Running Benchmark...")
    
    tts = VibeVoiceTTS()
    options = VibeVoiceOptions(speaker_name="Emma", ddpm_steps=5)
    
    # Test texts of varying lengths
    test_texts = [
        "Hello.",
        "This is a short test sentence.",
        "This is a longer sentence to test how the model handles more complex input with multiple clauses and phrases.",
        "Now we're testing an even longer piece of text. " * 5,
    ]
    
    results = []
    
    for text in test_texts:
        logger.info(f"Testing: {text[:50]}...")
        
        # Non-streaming benchmark
        start = time.perf_counter()
        sample_rate, audio = tts.tts(text, options)
        elapsed = time.perf_counter() - start
        duration = len(audio) / sample_rate
        rtf = elapsed / duration
        
        results.append({
            "chars": len(text),
            "duration": duration,
            "generation_time": elapsed,
            "rtf": rtf,
        })
        
        logger.info(f"  Audio: {duration:.2f}s | Gen time: {elapsed:.2f}s | RTF: {rtf:.2f}x")
    
    # Streaming latency test
    logger.info("\nStreaming Latency Test...")
    text = "This is a test of streaming latency."
    
    start = time.perf_counter()
    first_chunk_time = None
    total_samples = 0
    
    for sample_rate, chunk in tts.stream_tts_sync(text, options):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start
        total_samples += len(chunk)
    
    total_time = time.perf_counter() - start
    
    logger.info(f"  First chunk latency: {first_chunk_time*1000:.0f}ms")
    logger.info(f"  Total generation time: {total_time:.2f}s")
    logger.info(f"  Total audio duration: {total_samples/sample_rate:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="VibeVoice + FastRTC Examples")
    parser.add_argument(
        "--example",
        choices=["echo", "llm", "streaming", "multispeaker", "websocket", "benchmark"],
        default="echo",
        help="Which example to run",
    )
    
    args = parser.parse_args()
    
    examples = {
        "echo": example_echo_bot,
        "llm": example_llm_chat,
        "streaming": example_streaming_text,
        "multispeaker": example_multi_speaker,
        "websocket": example_websocket_server,
        "benchmark": example_benchmark,
    }
    
    examples[args.example]()


if __name__ == "__main__":
    main()
