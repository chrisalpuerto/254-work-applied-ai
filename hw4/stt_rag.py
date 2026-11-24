"""
CPSC 254 - Assignment 4, Part 4 (1): Speech-to-text -> syllabus RAG

Capture or load a spoken question, transcribe it, send the transcription to the
Part 3 RAG application, and display both the transcription and the answer.

Usage:
    python stt_rag.py --audio part4-audio/question_grading_weights.wav
    python stt_rag.py --audio part4-audio/BRS_Speech_Priest_Prayer_Roomy.wav
    python stt_rag.py --mic --seconds 6                 # live microphone
    python stt_rag.py --audio file.wav --engine google  # Google Web Speech instead of Whisper
    python stt_rag.py --audio file.wav --whisper-model small

Each run is also logged to results/part4_stt_<audio name>.txt
"""

import argparse
import os
import sys
import time

from speech_utils import load_audio_16k, record_microphone, transcribe

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
DEFAULT_AUDIO = os.path.join(HERE, "part4-audio", "question_grading_weights.wav")


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def main():
    ap = argparse.ArgumentParser(description="Spoken question -> STT -> syllabus RAG")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--audio", help=f"audio file (WAV/AIFF/FLAC); default {os.path.relpath(DEFAULT_AUDIO, HERE)}")
    src.add_argument("--mic", action="store_true", help="record from the microphone instead")
    ap.add_argument("--seconds", type=float, default=6, help="microphone recording length")
    ap.add_argument("--engine", choices=["whisper", "google"], default="whisper")
    ap.add_argument("--whisper-model", default="base", help="tiny/base/small/medium/large")
    ap.add_argument("--show-chunks", action="store_true", help="print retrieved syllabus text")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = "mic" if args.mic else os.path.splitext(os.path.basename(args.audio or DEFAULT_AUDIO))[0]
    sys.stdout = Tee(os.path.join(RESULTS_DIR, f"part4_stt_{name}.txt"))

    print("=" * 78)
    print("Part 4 (1): Speech-to-text -> CPSC 254 syllabus RAG")
    print("=" * 78)

    # ---- 1. capture / load speech --------------------------------------
    path = None
    if args.mic:
        try:
            audio = record_microphone(args.seconds)
        except Exception as e:
            sys.exit(f"Microphone unavailable ({type(e).__name__}: {e}). "
                     f"Use --audio <file> instead.")
        print(f"Input: microphone, {args.seconds:.0f} s at 16 kHz")
    else:
        path = args.audio or DEFAULT_AUDIO
        if not os.path.exists(path):
            sys.exit(f"Audio file not found: {path}")
        audio, sr, dur = load_audio_16k(path)
        conv = "16 kHz" if sr == 16000 else f"{sr} Hz -> resampled to 16 kHz"
        print(f"Input: {os.path.relpath(path, HERE)} ({dur:.1f} s, {conv} mono)")

    # ---- 2. transcribe ----------------------------------------------------
    engine_desc = (f"OpenAI Whisper '{args.whisper_model}' (local, CPU)" if args.engine == "whisper"
                   else "Google Web Speech API (SpeechRecognition)")
    print(f"STT engine: {engine_desc}")
    text, stt_time = transcribe(args.engine, args.whisper_model, path=path, audio_16k=audio)
    print(f"\nTRANSCRIPTION ({stt_time:.1f}s):\n  \"{text}\"")
    if not text:
        sys.exit("\nNo speech recognized; nothing to send to the RAG application.")

    # ---- 3. send to the Part 3 RAG application ------------------------------
    from rag import SyllabusRAG, print_result

    print()
    rag = SyllabusRAG()  # reuses the saved Chroma index from Part 3
    t0 = time.perf_counter()
    res = rag.ask(text)
    print("\nRAG RESULT:")
    print_result(res, args.show_chunks)
    if res["insufficient_info"]:
        print("\n(The transcribed speech is not answerable from the syllabus, so the RAG "
              "application declined instead of inventing an answer.)")
    print(f"\nTiming: STT {stt_time:.1f}s + RAG {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
