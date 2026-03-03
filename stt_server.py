"""
Demerzel STT Server
===================
Backend: whisper.cpp (Vulkan GPU) via HTTP API
VAD: Silero VAD (Python/PyTorch, 1 thread)
Audio: PyAudio mic capture, 48kHz → 16kHz downsampling
Location: ~/faster-whisper/ on okDemerzel

Architecture:
  Python (this server): mic capture → VAD gate → accumulate speech
  whisper.cpp (separate process): GPU transcription via HTTP on port 8891
  This server does NOT run Whisper inference — it delegates to whisper.cpp.

Prerequisites:
  whisper.cpp server must be running: ~/whisper-cpp/build/bin/whisper-server
"""

import os
import io
import wave
import time
import threading
import argparse
from collections import deque
from flask import Flask, jsonify, Response
import logging
import queue
import sys
import urllib.request
import json

# Set thread count BEFORE importing numpy/torch
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import torch
import pyaudio
from transcript_manager import TranscriptManager
from flask_socketio import SocketIO

# Filter noisy /transcript polling from logs
class TranscriptFilter(logging.Filter):
    def filter(self, record):
        return '/transcript' not in record.getMessage()

# Global transcription state
transcript_manager = TranscriptManager(max_history=50)

# Flask app
app = Flask(__name__, static_folder=".", static_url_path="")
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Audio capture constants ---
SAMPLE_RATE = 16000           # Target rate for Whisper
HW_SAMPLE_RATE = 48000        # Hardware native rate
HW_CHANNELS = 1               # USB mic is mono; onboard SN6186 is stereo (2)
PREFERRED_DEVICE_NAME = "USB"  # Auto-select USB mic if available
CHUNK = 4096
FORMAT = pyaudio.paInt16

# --- VAD constants ---
VAD_THRESHOLD = 0.4            # Speech probability threshold (Silero VAD)
SILENCE_CHUNKS_COMPLETE = 3    # ~0.3s — finalize quickly if sentence looks complete
SILENCE_CHUNKS_INCOMPLETE = 15 # ~1.5s — wait longer if mid-sentence
SPEECH_PAD_CHUNKS = 3          # Keep N chunks of audio before speech onset

# --- whisper.cpp server ---
WHISPER_CPP_URL = "http://localhost:8891"

# Global state
whisper_cpp_url = WHISPER_CPP_URL
vad_model = None
audio_queue = queue.Queue()
transcription_thread_running = True
language = "en"


def initialize_vad():
    """Initialize Silero VAD model. Uses 1 thread, <1ms per chunk."""
    global vad_model
    torch.set_num_threads(1)
    vad_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True
    )
    print("[OK] Silero VAD loaded (1 thread)")


def audio_to_wav_bytes(audio_np, sample_rate=SAMPLE_RATE):
    """Convert float32 numpy array to WAV bytes for whisper.cpp API."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()


def transcribe_via_whisper_cpp(audio_np):
    """Send audio to whisper.cpp HTTP API and return transcribed text."""
    wav_data = audio_to_wav_bytes(audio_np)
    boundary = b"----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n" +
        wav_data + b"\r\n"
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        b"json\r\n"
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="temperature"\r\n\r\n'
        b"0.0\r\n"
        b"--" + boundary + b"--\r\n"
    )
    req = urllib.request.Request(
        f"{whisper_cpp_url}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()).get("text", "").strip()
    except Exception as e:
        logging.error(f"whisper.cpp API error: {e}")
        return ""


def looks_complete(text):
    """Check if text looks like a complete sentence (ends with terminal punctuation)."""
    if not text:
        return False
    return text.rstrip()[-1] in '.?!'


def check_vad(audio_chunk):
    """Run Silero VAD on a 16kHz float32 audio chunk. Returns speech probability."""
    if len(audio_chunk) < 512:
        return 0.0
    # Silero VAD at 16kHz requires EXACTLY 512 samples per call
    tensor = torch.from_numpy(audio_chunk[-512:])
    prob = vad_model(tensor, SAMPLE_RATE).item()
    return prob


def transcribe_audio(socketio_app):
    """
    VAD-gated transcription loop.
    - Silero VAD checks every audio chunk (~0.5ms, near-zero CPU)
    - Only when speech is detected: accumulate audio
    - When speech ends (silence detected): transcribe the full utterance
    - Result: ~0.5% CPU idle, bursts only during speech
    """
    speech_buffer = []
    pre_speech_buffer = deque(maxlen=SPEECH_PAD_CHUNKS)
    is_speaking = False
    silence_counter = 0
    last_emit_text = ""

    while transcription_thread_running:
        try:
            # Get next audio chunk (blocks until available)
            try:
                audio_chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # --- VAD check (<1ms) ---
            speech_prob = check_vad(audio_chunk)

            if speech_prob > VAD_THRESHOLD:
                # Speech detected
                if not is_speaking:
                    # Speech onset — include pre-speech padding
                    is_speaking = True
                    speech_buffer = list(pre_speech_buffer)
                    print(f"[VAD] Speech started (prob={speech_prob:.2f})")
                silence_counter = 0
                speech_buffer.append(audio_chunk)

                # Live transcription during speech (every ~5 chunks)
                if len(speech_buffer) % 5 == 0 and len(speech_buffer) >= 5:
                    full_audio = np.concatenate(speech_buffer)
                    live_text = transcribe_via_whisper_cpp(full_audio)
                    if live_text and live_text != last_emit_text:
                        last_emit_text = live_text
                        transcript_manager.update_current_text(live_text)
                        socketio_app.emit('new_live_transcript', {'data': live_text})

            elif is_speaking:
                # Was speaking, now silent
                silence_counter += 1
                speech_buffer.append(audio_chunk)  # Keep tail padding

                # Hybrid endpointing: finalize faster if sentence looks complete
                if silence_counter >= SILENCE_CHUNKS_COMPLETE and looks_complete(last_emit_text):
                    threshold = SILENCE_CHUNKS_COMPLETE
                else:
                    threshold = SILENCE_CHUNKS_INCOMPLETE

                if silence_counter >= threshold:
                    # Speech ended — transcribe full utterance
                    mode = "complete" if threshold == SILENCE_CHUNKS_COMPLETE else "timeout"
                    print(f"[VAD] Speech ended ({len(speech_buffer)} chunks, {len(speech_buffer)*CHUNK/SAMPLE_RATE:.1f}s, {mode})")
                    full_audio = np.concatenate(speech_buffer)
                    final_text = transcribe_via_whisper_cpp(full_audio)

                    # Filter sound effects
                    stripped = final_text.strip()
                    if stripped and not (
                        (stripped.startswith('(') and stripped.endswith(')')) or
                        (stripped.startswith('[') and stripped.endswith(']'))
                    ):
                        print(f"[FINALIZED] {final_text}")
                        transcript_manager.update_current_text(final_text)
                        transcript_manager.finalize_text()
                        final_transcripts = transcript_manager.get_final_transcripts()
                        socketio_app.emit('new_final_transcript', {'data': final_transcripts})
                    else:
                        if stripped:
                            print(f"[FILTERED] {stripped}")
                        transcript_manager.update_current_text("")

                    socketio_app.emit('new_live_transcript', {'data': ''})
                    last_emit_text = ""

                    # Reset state
                    speech_buffer = []
                    is_speaking = False
                    silence_counter = 0
            else:
                # No speech — keep rolling pre-speech buffer (near-zero CPU)
                pre_speech_buffer.append(audio_chunk)

        except Exception as e:
            logging.error(f"Transcription error: {e}")
            socketio_app.sleep(0.1)


def downsample_to_mono16k(data_bytes, src_rate=HW_SAMPLE_RATE, src_channels=HW_CHANNELS):
    """Convert captured audio to mono 16kHz float32 for Whisper/VAD."""
    audio = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if src_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    ratio = src_rate // SAMPLE_RATE
    if ratio > 1:
        audio = audio[::ratio]
    return audio


def record_audio():
    """Record audio from microphone at native rate and downsample."""
    print("[AUDIO] Initializing PyAudio...")
    sys.stdout.flush()
    p = pyaudio.PyAudio()

    # Find the capture device — prefer USB mic
    device_index = None
    fallback_index = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"[AUDIO] Input device {i}: {info['name']} ({info['maxInputChannels']}ch, {int(info['defaultSampleRate'])}Hz)")
            if PREFERRED_DEVICE_NAME.lower() in info['name'].lower():
                device_index = i
            elif fallback_index is None and 'pipewire' not in info['name'].lower() and 'default' not in info['name'].lower():
                fallback_index = i

    if device_index is None:
        device_index = fallback_index

    if device_index is not None:
        dev_info = p.get_device_info_by_index(device_index)
        hw_channels = dev_info['maxInputChannels']
        print(f"[AUDIO] Using device {device_index}: {dev_info['name']} ({hw_channels}ch)")
    else:
        hw_channels = HW_CHANNELS
        print("[AUDIO] No input device found, using default")

    hw_chunk = CHUNK * (HW_SAMPLE_RATE // SAMPLE_RATE) * hw_channels
    print(f"[AUDIO] Capturing at {HW_SAMPLE_RATE}Hz, {hw_channels}ch, downsampling to {SAMPLE_RATE}Hz mono")

    try:
        stream = p.open(
            format=FORMAT,
            channels=hw_channels,
            rate=HW_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=hw_chunk,
        )
        print("[AUDIO] Recording started")
        sys.stdout.flush()

        while transcription_thread_running:
            try:
                data = stream.read(hw_chunk, exception_on_overflow=False)
                audio_np = downsample_to_mono16k(data, src_channels=hw_channels)
                audio_queue.put(audio_np)
            except Exception as e:
                print(f"[AUDIO] Error: {e}")
                break
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()


# --- Flask routes ---

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/transcript')
def get_transcript():
    return jsonify({
        'current': transcript_manager.get_current_text(),
        'final': transcript_manager.get_final_transcripts()
    })


@app.route('/stream')
def stream():
    def event_stream():
        last_sent = None
        while True:
            current = transcript_manager.get_current_text()
            if current != last_sent:
                last_sent = current
                yield f"data: {current}\n\n"
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")


@app.route('/clear', methods=['POST'])
def clear():
    transcript_manager.clear_all()
    socketio.emit('new_live_transcript', {'data': ''})
    socketio.emit('new_final_transcript', {'data': []})
    return jsonify({"status": "cleared"})


def start_transcription_service(whisper_url):
    """Start VAD, audio recording, and transcription threads."""
    global whisper_cpp_url
    whisper_cpp_url = whisper_url
    initialize_vad()
    print(f"[OK] Using whisper.cpp server at {whisper_url}")

    print("[INIT] Starting audio recording thread...")
    audio_thread = threading.Thread(target=record_audio, daemon=True)
    audio_thread.start()

    print("[INIT] Starting VAD-gated transcription thread...")
    transcription_thread = threading.Thread(target=transcribe_audio, args=(socketio,), daemon=True)
    transcription_thread.start()

    return audio_thread, transcription_thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demerzel STT Server (VAD + whisper.cpp GPU)")
    parser.add_argument("--port", type=int, default=8890, help="Web UI port (default: 8890)")
    parser.add_argument("--whisper-url", type=str, default=WHISPER_CPP_URL,
                        help=f"whisper.cpp server URL (default: {WHISPER_CPP_URL})")
    parser.add_argument("--vad-threshold", type=float, default=VAD_THRESHOLD,
                        help=f"VAD speech probability threshold (default: {VAD_THRESHOLD})")
    args = parser.parse_args()

    VAD_THRESHOLD = args.vad_threshold

    print(f"=== Demerzel STT Server ===")
    print(f"whisper.cpp backend: {args.whisper_url}")
    print(f"VAD threshold: {args.vad_threshold}")
    print(f"Web UI port: {args.port}")
    print(f"===========================")

    threads = start_transcription_service(args.whisper_url)

    log = logging.getLogger('werkzeug')
    log.addFilter(TranscriptFilter())

    try:
        socketio.run(app, host="0.0.0.0", port=args.port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        transcription_thread_running = False
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
