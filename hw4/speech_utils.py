"""
Shared speech helpers for Part 4 (stt_rag.py, tts_rag.py).

STT: OpenAI Whisper (local, as in the course example NLP_STT_whisper.py) or the
     Google Web Speech API via SpeechRecognition (course example NLP_STT_sr.py).
TTS: pyttsx3 (course example NLP_TTS.py), which uses the operating system's voices
     (NSSpeechSynthesizer on macOS, SAPI5 on Windows, eSpeak on Linux).
"""

import math
import os
import re
import sys
import time

import numpy as np

WHISPER_SR = 16_000  # Whisper expects 16 kHz mono float32


# ---------------------------------------------------------------------------
# Audio input
# ---------------------------------------------------------------------------
def load_audio_16k(path):
    """Read any WAV/AIFF/FLAC (8-32 bit, any rate, mono or stereo) and return
    16 kHz mono float32 in [-1, 1]. No ffmpeg needed."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    duration = len(audio) / sr
    audio = audio.mean(axis=1)                       # stereo -> mono
    if sr != WHISPER_SR:
        g = math.gcd(sr, WHISPER_SR)
        audio = resample_poly(audio, WHISPER_SR // g, sr // g).astype(np.float32)
    return audio, sr, duration


def record_microphone(seconds=6.0):
    """Record from the default microphone (course example approach)."""
    import sounddevice as sd

    print(f"Recording for {seconds:.0f} seconds... speak your question now.")
    audio = sd.rec(int(seconds * WHISPER_SR), samplerate=WHISPER_SR, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    audio = audio.flatten()
    peak = float(np.max(np.abs(audio))) or 1.0
    return audio / peak


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------
_whisper_cache = {}


def transcribe_whisper(audio_16k, model_name="base"):
    import whisper

    if model_name not in _whisper_cache:
        _whisper_cache[model_name] = whisper.load_model(model_name)
    result = _whisper_cache[model_name].transcribe(audio_16k, fp16=False, language="en")
    return result["text"].strip()


def transcribe_google(path=None, audio_16k=None):
    """Google Web Speech API (free endpoint, needs internet)."""
    import speech_recognition as sr

    rec = sr.Recognizer()
    if path is not None and audio_16k is None:
        audio_16k, _, _ = load_audio_16k(path)
    pcm16 = (np.clip(audio_16k, -1, 1) * 32767).astype("<i2").tobytes()
    data = sr.AudioData(pcm16, WHISPER_SR, 2)
    try:
        return rec.recognize_google(data)
    except sr.UnknownValueError:
        return ""


def transcribe(engine, model_name="base", path=None, audio_16k=None):
    t0 = time.perf_counter()
    if engine == "whisper":
        if audio_16k is None:
            audio_16k, _, _ = load_audio_16k(path)
        text = transcribe_whisper(audio_16k, model_name)
    elif engine == "google":
        text = transcribe_google(path, audio_16k)
    else:
        raise ValueError(f"unknown STT engine: {engine}")
    return text, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------
def clean_for_speech(text):
    """Turn a RAG answer into text that sounds natural when read aloud:
    drop citation brackets and markdown, expand a few symbols."""
    t = text
    t = re.sub(r"\[(?:Excerpt\s*\d+\s*[|,]?\s*)?(?:p(?:age|\.)?\s*[\d\-‐-―, ]+)\]", "", t, flags=re.I)
    t = re.sub(r"\[Excerpt[^\]]*\]", "", t, flags=re.I)
    t = re.sub(r"【[^】]*】", "", t)                     # full-width citation brackets
    t = t.replace("**", "").replace("__", "")
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)       # bullet markers
    t = re.sub(r"[‐-―−]", "-", t)          # unicode dashes / hyphens
    # ranges: "2:15 pm - 4:00 pm" / "8-11" -> "2:15 pm to 4:00 pm" / "8 to 11"
    t = re.sub(r"(\d(?:\s*[ap]\.?m\.?)?)\s*-\s*(\d)", r"\1 to \2", t, flags=re.I)
    t = re.sub(r"(\d)\s*%", r"\1 percent", t)
    t = t.replace(">=", "at least ").replace("&", " and ")
    # join lines; give bullet lines without end punctuation a pause
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    t = " ".join(ln if ln[-1] in ".!?;:," else ln + "." for ln in lines)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)              # "exams ." -> "exams."
    return re.sub(r"\s{2,}", " ", t).strip()


def default_tts_path(stem):
    # pyttsx3 on macOS (NSSpeechSynthesizer) always writes AIFF, whatever the extension
    ext = ".aiff" if sys.platform == "darwin" else ".wav"
    return stem + ext


def synthesize(text, out_path, rate=175, voice=None):
    """Synthesize `text` to an audio file with pyttsx3. Returns the voice name used."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    chosen = None
    if voice:
        for v in engine.getProperty("voices"):
            if voice.lower() in (v.name or "").lower() or voice.lower() in v.id.lower():
                engine.setProperty("voice", v.id)
                chosen = v.name
                break
    if chosen is None:
        current = engine.getProperty("voice")
        chosen = next((v.name for v in engine.getProperty("voices") if v.id == current), None)
        chosen = chosen or current or "system default"
    if os.path.exists(out_path):
        os.remove(out_path)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    # the native driver writes asynchronously on some platforms; wait for the file
    for _ in range(50):
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            break
        time.sleep(0.1)
    engine.stop()
    return chosen


def audio_duration(path):
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


def play_audio(path):
    """Play an audio file through the default output device and block until done."""
    try:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        sd.play(data, sr)
        sd.wait()
        return "sounddevice"
    except Exception as e:  # no output device, unsupported format, ...
        if sys.platform == "darwin" and os.system(f'afplay "{path}"') == 0:
            return "afplay"
        raise RuntimeError(f"audio playback unavailable ({type(e).__name__}: {e})") from e
