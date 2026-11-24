# Part 4: Speech Recognition and Voice Synthesis

Both programs extend the Part 3 application: they import `SyllabusRAG` from `rag.py` and reuse the saved Chroma index in `chroma_syllabus/`. The speech components follow the course examples:

- `NLP_STT_whisper.py`: OpenAI Whisper STT
- `NLP_STT_sr.py`: Google speech recognition
- `NLP_TTS.py`: pyttsx3 TTS

Shared helpers are in `speech_utils.py`.

| Component | Choice | Notes |
|---|---|---|
| Audio input | Audio file (`--audio`) or microphone (`--mic`, 16 kHz, `sounddevice`) | Files are read with `soundfile` (any bit depth, WAV/AIFF/FLAC), averaged to mono, and resampled to 16 kHz with `scipy.signal.resample_poly`. No ffmpeg needed. |
| STT | **OpenAI Whisper `base`** (74M parameters, local CPU, `fp16=False`, `language="en"`) | Optional `--engine google` uses the Google Web Speech API via SpeechRecognition (needs internet). |
| RAG | Part 3 `SyllabusRAG.ask()`: Chroma top-4 retrieval → gpt-oss | Unchanged, including the "not enough information" safeguard. |
| Speech cleanup | `clean_for_speech()` | Removes page citations (`[p. 3]`, `【p. 3】`) and markdown bullets/bold. Rewrites "%" as "percent" and "2:15 pm – 4:00 pm" as "2:15 pm to 4:00 pm" so the voice reads it naturally. |
| TTS | **pyttsx3** (macOS system voice via NSSpeechSynthesizer), 175 wpm | Saved to `results/tts_answer.aiff`. On macOS pyttsx3 always writes AIFF; on Windows/Linux it writes `.wav`. |
| Playback | `sounddevice` (fallback: macOS `afplay`) | Blocks until the audio finishes. `--no-play` skips playback. |
| Verification | `tts_rag.py --verify` | Re-transcribes the generated speech with Whisper, so a screenshot can show what was actually spoken. |

## Test audio

| File | Content | Purpose |
|---|---|---|
| `part4-audio/BRS_Speech_Priest_Prayer_Roomy.wav` (provided) | 29.6 s, 44.1 kHz, 24-bit mono recording of a prayer read in a reverberant room | Tests STT on real, non-studio speech, and tests that the RAG refuses an off-topic input instead of hallucinating. |
| `part4-audio/question_grading_weights.wav` (generated with macOS `say`, 16 kHz/16-bit) | "How is the final grade weighted between participation, assignments, and exams?" | A syllabus question, to show a complete answer from spoken input. |

## (1) Speech-to-text: `stt_rag.py`

**Run 1, provided recording:** `python stt_rag.py --audio part4-audio/BRS_Speech_Priest_Prayer_Roomy.wav`

```
Input: part4-audio/BRS_Speech_Priest_Prayer_Roomy.wav (29.6 s, 44100 Hz -> resampled to 16 kHz mono)
STT engine: OpenAI Whisper 'base' (local, CPU)

TRANSCRIPTION (2.0s):
  "We pray to thee our God this day, for the blessings you provide, but from ashes to ashes and
   from dust to dust so it must be. But keep this woman, O King of kings, be Nethial mighty wings.
   O God, the Spirit of all our joys, the cause of our delights, the glory of our darkest nights,
   and the comfort at times like these."

RAG RESULT:
A: The syllabus does not contain enough information to answer this question. The syllabus provides
   details about the course topics, schedule, emergency policy, and technical support for CPSC 254.
   Retrieved: p.5 (dist 0.680), p.4-5 (dist 0.688), p.4 (dist 0.690), p.5 (dist 0.698)   [6.5s]

(The transcribed speech is not answerable from the syllabus, so the RAG application declined
 instead of inventing an answer.)
Timing: STT 2.0s + RAG 6.5s
```

- **Transcription quality:** Whisper transcribed 30 s of reverberant speech in 2 s on the CPU. It is accurate apart from one likely error: "be Nethial mighty wings" is probably "beneath Thy mighty wings", an archaic phrase that the small `base` model misheard.
- **RAG behaviour:** the prayer is not a question about the course. Retrieval distances were all ≈ 0.68–0.70, compared with 0.10–0.38 for real syllabus questions in Part 3, and the RAG correctly returned the "not enough information" response.

**Run 2, spoken syllabus question:** `python stt_rag.py --audio part4-audio/question_grading_weights.wav`

```
TRANSCRIPTION (1.1s):
  "How is the final grade weighted between participation, assignments, and exams?"

A: The final grade is weighted as follows: participation and attendance 7 %, assignments 43 %,
   and exams 50 % (total 100 %)【p. 3】.
   Retrieved: p.3 (chunk 11, dist 0.198), p.2 (dist 0.365), p.3 (dist 0.394), p.2 (dist 0.424)   [4.5s]
Timing: STT 1.1s + RAG 4.5s
```

The transcription is word-for-word correct and the answer matches the syllabus Grading Policy table (p. 3). As a cross-check, `--engine google` on the same file returned "how is the final grade weighted between participation assignments and exam". That is lowercase, unpunctuated, and missing the plural, which shows why Whisper is the default.

## (2) Text-to-speech: `tts_rag.py`

**Run 3, typed question, spoken answer:** `python tts_rag.py --verify`. The default question is "What are the instructor's office hours and where is the office?"

```
RAG ANSWER (5.3s, retrieved p.1, p.4, p.4-5, p.4):
Instructor office hours are Monday and Wednesday 2:15 pm – 4:00 pm and 5:15 pm – 5:30 pm, or by
Zoom appointment; the instructor's office is in room CS-421. [p. 1]

TEXT SENT TO TTS (citations/markdown removed):
Instructor office hours are Monday and Wednesday 2:15 pm to 4:00 pm and 5:15 pm to 5:30 pm, or by
Zoom appointment; the instructor's office is in room CS-421.

[TTS] pyttsx3, voice 'system default', rate 175 wpm -> results/tts_answer.aiff (11.8 s of audio, 514 KB, 2.5s)
[PLAY] playing the spoken answer...
[PLAY] finished (via sounddevice)

[VERIFY] Whisper transcription of the generated speech:
  "Instructor office hours are Monday and Wednesday 2 15pm to 4pm and 515pm to 530pm or by Zoom
   appointment. The instructor's office is in room CS 421."
```

**Run 4, full voice loop (spoken question → spoken answer):** `python tts_rag.py --from-audio part4-audio/question_grading_weights.wav --verify`

```
[STT] part4-audio/question_grading_weights.wav -> "How is the final grade weighted between participation, assignments, and exams?"
RAG ANSWER: The final grade is weighted 7 % for class participation and attendance, 43 % for
            assignments, and 50 % for exams [p. 3].
TEXT SENT TO TTS: The final grade is weighted 7 percent for class participation and attendance,
            43 percent for assignments, and 50 percent for exams .
[TTS] -> results/tts_answer.aiff (7.9 s of audio)   (run with --no-play)
[VERIFY] "The final grade is weighted 7% for class participation and attendance, 43% for
          assignments, and 50% for exams."
```

The Whisper round trip confirms that the audio file contains the correct spoken answer. Without the speech cleanup, the first version read "2:15 pm – 4:00 pm" as "2 15 pm 4 pm", which loses the meaning of the range. The cleanup step fixed it.

## Evidence

*Insert screenshots of the terminal running the commands above.* The logs of the runs are saved as text:

- `results/part4_stt_BRS_Speech_Priest_Prayer_Roomy.txt`
- `results/part4_stt_question_grading_weights.txt`
- `results/part4_tts_output.txt` (the last run)
- `results/tts_answer.aiff`, the spoken answer

A screenshot cannot show sound, so the `[PLAY] finished` line together with the `[VERIFY]` transcription is the evidence that the spoken answer was produced and played.

## Limitations

- **Microphone:** microphone mode (`--mic`) is implemented following the course example but was not used for the reported runs. File input is reproducible and was available. The microphone needs macOS permission for the terminal app.
- **Whisper `base`:** it is fast on the CPU (1–2 s here) but makes occasional errors on archaic or unusual words ("be Nethial"). `--whisper-model small` is more accurate but slower.
- **Off-topic input:** when the spoken input is not a question (like the provided prayer), the pipeline still runs end to end. The safeguard is the RAG's refusal rule; there is no separate "is this a question?" check.
- **Voice quality:** pyttsx3 uses the operating system's offline voices, which sound robotic compared with neural TTS. On macOS it can only write AIFF.
- **Latency:** a voice-to-voice answer takes about 7–10 s (STT ~1–2 s, RAG ~3–6 s, TTS ~2 s), plus the audio's own playback time.
