# Demerzel STT

Speech-to-text server for [Demerzel](https://github.com/dan-gearscodeandfire), built around Silero VAD and whisper.cpp with Vulkan GPU acceleration.

## Architecture

```
Microphone (USB, 48kHz mono)
    |
    v
Python server (stt_server.py, port 8890)
    +-- Downsample 48kHz -> 16kHz
    +-- Silero VAD gate (<1ms per chunk)
    +-- Accumulate speech buffer
    +-- Hybrid endpointing
    |       +-- Complete sentence (. ? !) -> finalize in ~0.3s
    |       +-- Incomplete sentence -> wait up to ~1.5s
    +-- POST WAV to whisper.cpp
            |
            v
whisper.cpp server (port 8891, Vulkan GPU)
    +-- GGML base.en model
    +-- AMD Radeon 8060S (RDNA 3.5, 40 CUs)
    +-- Returns transcription (~100-300ms)
            |
            v
Flask web UI + SocketIO (real-time updates)
```

## Files

| File | Purpose |
|------|---------|
| `stt_server.py` | Main server — mic capture, Silero VAD, hybrid endpointing, Flask web UI |
| `transcript_manager.py` | Intelligent transcript deduplication via `difflib` similarity matching |
| `index.html` | Dark-themed web UI — shows live, staged, and finalized transcripts |

## Prerequisites

- Python 3.12+ with venv
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) built with Vulkan support
- USB microphone
- PyAudio, torch, numpy, flask, flask-socketio

## Quick Start

1. Start whisper.cpp server:
   ```bash
   cd ~/whisper-cpp
   ./build/bin/whisper-server -m models/ggml-base.en.bin --host 0.0.0.0 --port 8891 -t 4 -l en
   ```

2. Start the STT server:
   ```bash
   cd ~/faster-whisper
   source .venv/bin/activate
   python stt_server.py
   ```

3. Open the web UI at `http://<host>:8890`

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `VAD_THRESHOLD` | 0.4 | Speech probability threshold (Silero VAD) |
| `SILENCE_CHUNKS_COMPLETE` | 3 (~0.3s) | Silence before finalizing a complete sentence |
| `SILENCE_CHUNKS_INCOMPLETE` | 15 (~1.5s) | Silence before finalizing an incomplete sentence |
| `SPEECH_PAD_CHUNKS` | 3 | Audio chunks kept before speech onset for context |
| `WHISPER_CPP_URL` | http://localhost:8891 | whisper.cpp server endpoint |

## Hybrid Endpointing

Instead of a fixed silence timeout, the server uses two thresholds based on whether Whisper's transcription looks like a complete sentence:

- **Complete** (ends with `. ? !`): finalize after ~0.3s of silence
- **Incomplete** (no terminal punctuation): wait up to ~1.5s before timing out

This means natural sentence endings get near-instant finalization, while mid-sentence pauses (breathing, thinking) don't prematurely cut off speech.

## Performance (okDemerzel)

| Metric | Value |
|--------|-------|
| Hardware | AMD Ryzen AI Max+ 395, Radeon 8060S |
| CPU idle | <1% |
| CPU during speech | ~10% (VAD + HTTP overhead only) |
| Inference latency | ~100-300ms per utterance |
| Python process memory | ~585MB |
| whisper-server memory | ~229MB |

## Changes from Little Timmy (stt-server-v17)

This server evolved from the [Little Timmy STT server](https://github.com/dan-gearscodeandfire/little_timmy/tree/main/stt-server-v17) (`timmy_hears.py`). Key changes:

### Inference Engine
- **Before**: faster-whisper (CTranslate2), CUDA GPU with CPU fallback, all in one Python process
- **After**: whisper.cpp (GGML) via HTTP API, Vulkan GPU on a separate process. Python handles only mic capture and VAD.

### Voice Activity Detection
- **Before**: None — continuous transcription at 0.1s intervals, constant high CPU usage
- **After**: Silero VAD gate — Whisper only runs during actual speech, <1% CPU when idle

### Endpointing
- **Before**: Fixed 0.5s silence threshold, caused premature finalizations on short pauses
- **After**: Hybrid endpointing — 0.3s for complete sentences, 1.5s for incomplete. Punctuation-based semantic detection.

### Model and Tuning
- **Before**: `small_dan_ct2` (248MB custom model), beam_size=5, default threads
- **After**: `ggml-base.en` (142MB), beam_size=1, greedy decode, int8, 4 threads. Custom GGML voice model planned.

### Operating Modes
- **Before**: Two modes (`--ai` for LLM routing, default for TTS), echo cancellation, distributed pipeline (LLM preprocessor on port 5050, TTS on port 5051, ESP32 eye display)
- **After**: Single-purpose STT server. LLM routing and TTS are handled separately by Demerzel.

### Web UI
- **Before**: Port 8888, polled `/transcript` every 500ms
- **After**: Port 8890, Flask-SocketIO for real-time push updates, dark theme

### Transcript Manager
- **Unchanged**: `transcript_manager.py` carries over from Little Timmy — same `difflib`-based deduplication logic.

## Port Assignments (okDemerzel)

| Service | Port |
|---------|------|
| llama.cpp (LLM) | 8080 |
| SearXNG | 8888 |
| STT server (this) | 8890 |
| whisper.cpp (GPU) | 8891 |

## Note on Directory Name

This repo lives at `~/faster-whisper/` on okDemerzel — a legacy name from when it used the faster-whisper library. It now delegates all inference to whisper.cpp. The name was kept to avoid updating references across code, documentation, and memory systems.
