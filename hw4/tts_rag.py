"""
CPSC 254 - Assignment 4, Part 4 (2): syllabus RAG -> text-to-speech

Send a question through the Part 3 RAG application, convert the text answer to
speech with pyttsx3, save it, and play it aloud.

Usage:
    python tts_rag.py "What are the instructor's office hours?"
    python tts_rag.py                                   # uses a default sample question
    python tts_rag.py --from-audio part4-audio/question_grading_weights.wav   # voice in, voice out
    python tts_rag.py "question" --no-play              # save audio without playing
    python tts_rag.py "question" --verify               # re-transcribe the spoken answer with Whisper
    python tts_rag.py "question" --voice Samantha --rate 170

Outputs: results/tts_answer.aiff (macOS) or .wav, and results/part4_tts_output.txt
"""

import argparse
import os
import sys
import time

from speech_utils import (audio_duration, clean_for_speech, default_tts_path, play_audio,
                          synthesize, transcribe)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
DEFAULT_QUESTION = "What are the instructor's office hours and where is the office?"


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
    ap = argparse.ArgumentParser(description="Syllabus RAG answer -> spoken audio")
    ap.add_argument("question", nargs="*", help="question text")
    ap.add_argument("--from-audio", help="take the question from an audio file (STT first)")
    ap.add_argument("--out", help="output audio path (default results/tts_answer.aiff|.wav)")
    ap.add_argument("--voice", help="voice name substring, e.g. Samantha, Daniel, Zira")
    ap.add_argument("--rate", type=int, default=175, help="speech rate, words per minute")
    ap.add_argument("--no-play", action="store_true", help="do not play the audio")
    ap.add_argument("--verify", action="store_true",
                    help="transcribe the generated speech with Whisper to confirm its content")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS_DIR, "part4_tts_output.txt"))
    out_path = args.out or default_tts_path(os.path.join(RESULTS_DIR, "tts_answer"))

    print("=" * 78)
    print("Part 4 (2): CPSC 254 syllabus RAG -> text-to-speech")
    print("=" * 78)

    # ---- 1. question (typed, or spoken and transcribed) ---------------------
    if args.from_audio:
        q, t = transcribe("whisper", path=args.from_audio)
        print(f"[STT] {os.path.relpath(args.from_audio, HERE)} -> \"{q}\" ({t:.1f}s)")
    else:
        q = " ".join(args.question) or DEFAULT_QUESTION
    print(f"\nQUESTION: {q}\n")

    # ---- 2. RAG answer ------------------------------------------------------
    from rag import SyllabusRAG

    rag = SyllabusRAG()  # reuses the saved Chroma index from Part 3
    res = rag.ask(q)
    pages = ", ".join(f"p.{s['pages']}" for s in res["sources"])
    print(f"\nRAG ANSWER ({res['latency_s']:.1f}s, retrieved {pages}):\n{res['answer']}")

    # ---- 3. text -> speech ---------------------------------------------------
    spoken = clean_for_speech(res["answer"])
    print(f"\nTEXT SENT TO TTS (citations/markdown removed):\n{spoken}")
    t0 = time.perf_counter()
    voice = synthesize(spoken, out_path, rate=args.rate, voice=args.voice)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        sys.exit("TTS failed: no audio file was produced.")
    print(f"\n[TTS] pyttsx3, voice '{voice}', rate {args.rate} wpm -> "
          f"{os.path.relpath(out_path, HERE)} ({audio_duration(out_path):.1f} s of audio, "
          f"{os.path.getsize(out_path) / 1024:.0f} KB, {time.perf_counter() - t0:.1f}s)")

    # ---- 4. play aloud ---------------------------------------------------------
    if args.no_play:
        print("[PLAY] skipped (--no-play)")
    else:
        try:
            print("[PLAY] playing the spoken answer...")
            how = play_audio(out_path)
            print(f"[PLAY] finished (via {how})")
        except RuntimeError as e:
            print(f"[PLAY] {e}. The answer audio is saved at {out_path}.")

    # ---- 5. optional check: what does the audio actually say? ----------------
    if args.verify:
        heard, t = transcribe("whisper", path=out_path)
        print(f"\n[VERIFY] Whisper transcription of the generated speech ({t:.1f}s):\n  \"{heard}\"")


if __name__ == "__main__":
    main()
