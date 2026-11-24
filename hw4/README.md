# CPSC 254 – Assignment 4

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 on macOS (Apple Silicon, 10 CPU cores, 24 GB RAM). No local GPU is needed (Whisper `base` runs on the CPU).

### API access (Parts 2–3)

Parts 2 and 3 call the NRP Nautilus LLM service (OpenAI-compatible API at `https://ellm.nrp-nautilus.io/v1`):

- chat model `gpt-oss` (served as `openai/gpt-oss-120b`)
- embedding model `qwen3-embedding`

Put your key in a `.env` file in this folder or any parent folder. Do **not** submit this file.

```
OPENAI_API_KEY=your-nautilus-token
```

Optional overrides: `NRP_BASE_URL`, `NRP_CHAT_MODEL`, `NRP_EMBED_MODEL`. Connection settings live in `nrp_llm.py`.

## Programs

| Part | Command | What it does |
|---|---|---|
| 1 | `python topic_classifier.py` | Vectorizes AG News, trains and evaluates the MLP for 4 configurations (~4–5 min). Add `--quick` for a 10-second smoke test. |
| 2 | `python classification_agent.py "article text"` | Classifies one article and prints `{"category": ..., "justification": ...}` |
| 2 | `python classification_agent.py --show-prompt` | Prints the zero-shot prompt template |
| 2 | `python classification_agent.py --eval` | Agent vs. best Part 1 model on a stratified 200-example test sample (seed 42), ~10 min |
| 3 | `python rag.py --build` | Builds the Chroma index of `CPSC 254_SYL.pdf` in `chroma_syllabus/` (reused on later runs) |
| 3 | `python rag.py "question"` or `python rag.py` | Answers one question, or starts an interactive loop. `--show-chunks` prints the retrieved text; `--rebuild` forces re-indexing. |
| 3 | `python rag_eval.py` | Loads the saved index and evaluates 6 syllabus questions |
| 4 | `python stt_rag.py --audio part4-audio/question_grading_weights.wav` | Transcribes a spoken question (Whisper `base`) and answers it with the RAG app. `--mic --seconds 6` records from the microphone instead; `--engine google` uses Google Web Speech. |
| 4 | `python stt_rag.py --audio part4-audio/BRS_Speech_Priest_Prayer_Roomy.wav` | Provided test recording (a prayer, not a syllabus question). The RAG app correctly answers "not enough information". |
| 4 | `python tts_rag.py "question"` | RAG answer → pyttsx3 speech → saved to `results/tts_answer.aiff` and played aloud. `--verify` re-transcribes the audio; `--no-play` saves without playing. |
| 4 | `python tts_rag.py --from-audio part4-audio/question_grading_weights.wav --verify` | Full voice loop: spoken question → STT → RAG → spoken answer |

**Data:**

- `data/train.jsonl` and `data/test.jsonl` hold the official AG News split. torchtext does not install on Python ≥ 3.13, so these files are used as the fallback.
- `CPSC 254_SYL.pdf` (from Canvas) must be in this folder for Part 3.
- Part 4 reuses the Part 3 index, so run `python rag.py --build` first.
- `part4-audio/question_grading_weights.wav` was generated with macOS `say` (`say -o part4-audio/question_grading_weights.wav --data-format=LEI16@16000 "How is the final grade weighted between participation, assignments, and exams?"`).
- Whisper reads audio through `soundfile`, so ffmpeg is not required.
- On macOS, pyttsx3 always writes AIFF, so the answer file is `.aiff`; on Windows/Linux it is `.wav`.

**Outputs** go to `results/`:

| Part | Files |
|---|---|
| 1 | `part1_output.txt`, `part1_summary.*`, `confusion_*.png`, `misclassified_examples.txt` |
| 2 | `part2_output.txt`, `part2_predictions.csv`, `part2_comparison.md`. The Part 1 model is cached as `part1_best_model.joblib` (regenerated automatically if missing). |
| 3 | `part3_eval_output.txt`, `part3_eval.json` |
| 4 | `part4_stt_<audio>.txt`, `part4_tts_output.txt`, `tts_answer.aiff` |

Written answers are in `part1_report.md`, `part2_3_report.md`, and `part4_report.md`.
