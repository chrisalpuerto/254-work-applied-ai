# CPSC 254 – Assignment 4

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13 on macOS (Apple Silicon, 10 CPU cores, 24 GB RAM). No local GPU is needed.

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

**Data:**

- `data/train.jsonl` and `data/test.jsonl` hold the official AG News split. torchtext does not install on Python ≥ 3.13, so these files are used as the fallback.
- `CPSC 254_SYL.pdf` (from Canvas) must be in this folder for Part 3.

**Outputs** go to `results/`:

| Part | Files |
|---|---|
| 1 | `part1_output.txt`, `part1_summary.*`, `confusion_*.png`, `misclassified_examples.txt` |
| 2 | `part2_output.txt`, `part2_predictions.csv`, `part2_comparison.md`. The Part 1 model is cached as `part1_best_model.joblib` (regenerated automatically if missing). |
| 3 | `part3_eval_output.txt`, `part3_eval.json` |

Written answers are in `part1_report.md` and `part2_3_report.md`.
