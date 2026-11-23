"""
CPSC 254 - Assignment 4, Part 2
LLM-based zero-shot news topic classification agent (LangChain + gpt-oss on NRP Nautilus).

Usage:
    # (1) classify one article
    python classification_agent.py "Intel posts record profit as chip demand soars ..."
    echo "article text" | python classification_agent.py

    # (2) compare with the best Part 1 classifier on a stratified 200-example test sample
    python classification_agent.py --eval            # n=200, seed=42
    python classification_agent.py --eval --n 50     # smaller sample if needed

Outputs of --eval (in ./results/):
    part2_output.txt, part2_predictions.csv, part2_comparison.md
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from nrp_llm import CHAT_MODEL, NRP_BASE_URL, get_chat_model

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_TO_NAME = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
NAME_TO_LABEL = {v: k for k, v in LABEL_TO_NAME.items()}
MAX_JUSTIFICATION_WORDS = 40

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PART1_MODEL_PATH = os.path.join(RESULTS_DIR, "part1_best_model.joblib")

# ---------------------------------------------------------------------------
# (1b) Zero-shot prompt template (no labeled examples are given to the model)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a news topic classification agent for the AG News dataset.
Classify the news article given by the user into exactly ONE of these four categories:

- World: international and national affairs, politics, government, elections, war and conflict, diplomacy, crime, disasters
- Sports: sports events, athletes, teams, games, leagues, tournaments
- Business: companies, markets, the economy, finance, earnings, trade, industry, labor
- Sci/Tech: science, technology, computing, the internet, software, space, scientific and medical research

Respond with ONLY a single JSON object and nothing else (no markdown, no code fences),
with exactly these two keys:
{"category": "<one of: World, Sports, Business, Sci/Tech>", "justification": "<one short sentence, at most 30 words, naming the key cues in the article>"}

Do not include step-by-step reasoning. The justification must be a concise explanation only."""

USER_TEMPLATE = "Article:\n{article}"

RETRY_TEMPLATE = (
    "Your previous reply was invalid: {error}\n"
    "Reply again with ONLY the JSON object containing exactly the keys "
    '"category" (one of World, Sports, Business, Sci/Tech) and "justification".'
)


# ---------------------------------------------------------------------------
# (1c-d) Output schema and validation
# ---------------------------------------------------------------------------
class TopicClassification(BaseModel):
    """Validated agent output: exactly two fields, category must be a permitted name."""

    model_config = ConfigDict(extra="forbid")  # reject any extra keys

    category: Literal["World", "Sports", "Business", "Sci/Tech"]
    justification: str = Field(min_length=1)

    @field_validator("justification")
    @classmethod
    def brief(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("justification is empty")
        if len(v.split()) > MAX_JUSTIFICATION_WORDS:
            raise ValueError(f"justification longer than {MAX_JUSTIFICATION_WORDS} words")
        return v


def parse_and_validate(raw: str) -> TopicClassification:
    """Parse the model's text as JSON and validate it. Raises ValueError on failure."""
    text = raw.strip()
    # tolerate a markdown code fence around the JSON, nothing else
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"output is not valid JSON ({e.msg})") from None
    if not isinstance(data, dict):
        raise ValueError("output JSON is not an object")
    # normalize harmless case/spacing differences, e.g. "sci/tech " -> "Sci/Tech"
    if isinstance(data.get("category"), str):
        lookup = {c.lower(): c for c in CATEGORIES}
        data["category"] = lookup.get(data["category"].strip().lower(), data["category"])
    try:
        return TopicClassification.model_validate(data)
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'object'}: {err['msg']}"
                         for err in e.errors())
        raise ValueError(msgs) from None


# ---------------------------------------------------------------------------
# (1a) The agent
# ---------------------------------------------------------------------------
class ClassificationAgent:
    """Zero-shot topic classifier built with LangChain's create_agent."""

    def __init__(self, reasoning_effort="low"):
        from langchain.agents import create_agent

        llm = get_chat_model(temperature=0.0, reasoning_effort=reasoning_effort)
        # No tools are needed: the agent's only job is to read and label the article.
        self.agent = create_agent(model=llm, tools=[], system_prompt=SYSTEM_PROMPT)

    def _call(self, messages):
        result = self.agent.invoke({"messages": messages})
        msg = result["messages"][-1]
        usage = getattr(msg, "usage_metadata", None) or {}
        return msg.content if isinstance(msg.content, str) else str(msg.content), usage

    def classify(self, article: str) -> dict:
        """
        Returns a dict:
          {"category": ..., "justification": ...}                       on success
          {"error": "...", "raw_output": "..."}                          on failure
        plus bookkeeping keys under "_meta" (attempts, latency, tokens).
        """
        t0 = time.perf_counter()
        messages = [("user", USER_TEMPLATE.format(article=article.strip()))]
        tokens = {"input": 0, "output": 0, "reasoning": 0}
        last_error, raw = None, ""

        for attempt in (1, 2):  # first try + one retry
            try:
                raw, usage = self._call(messages)
            except Exception as e:  # network / server errors
                last_error, raw = f"LLM call failed: {type(e).__name__}: {e}", ""
                continue
            tokens["input"] += usage.get("input_tokens", 0)
            tokens["output"] += usage.get("output_tokens", 0)
            tokens["reasoning"] += (usage.get("output_token_details") or {}).get("reasoning", 0)
            try:
                parsed = parse_and_validate(raw)
                return {**parsed.model_dump(),
                        "_meta": {"attempts": attempt, "latency_s": time.perf_counter() - t0,
                                  "tokens": tokens}}
            except ValueError as e:
                last_error = str(e)
                # ask once more, showing the model its own invalid reply and the error
                messages = messages + [("assistant", raw),
                                       ("user", RETRY_TEMPLATE.format(error=last_error))]

        return {"error": f"Validation failed after retry: {last_error}",
                "raw_output": raw,
                "_meta": {"attempts": 2, "latency_s": time.perf_counter() - t0, "tokens": tokens}}


def public_json(result: dict) -> str:
    """The JSON the agent returns to callers (bookkeeping removed)."""
    return json.dumps({k: v for k, v in result.items() if k != "_meta"}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# (2) Comparison with the best Part 1 classifier
# ---------------------------------------------------------------------------
def load_part1_model():
    """Best Part 1 configuration: TF-IDF (no stop-word removal) + the same MLP.
    Trained on the full 120k training set only; cached after the first run."""
    import joblib

    if os.path.exists(PART1_MODEL_PATH):
        return joblib.load(PART1_MODEL_PATH), "loaded from cache"

    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline

    from topic_classifier import MLP_PARAMS, load_data, make_vectorizer

    train_df, _, _ = load_data()
    pipe = make_pipeline(make_vectorizer("Tfidf", stop_words=False), MLPClassifier(**MLP_PARAMS))
    print("Training the Part 1 model (TF-IDF, no stop-word removal, MLP 128 ReLU, 5 epochs)...")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipe.fit(train_df["text"], train_df["label"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(pipe, PART1_MODEL_PATH)
    return pipe, f"trained in {time.time() - t0:.0f}s and cached"


def run_eval(n, seed):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
    from sklearn.model_selection import train_test_split

    from topic_classifier import Tee, load_data

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS_DIR, "part2_output.txt"))

    _, test_df, source = load_data()
    # Stratified sample from the OFFICIAL TEST SET, seed 42 (never used for training/tuning)
    sample, _ = train_test_split(test_df, train_size=n, stratify=test_df["label"],
                                 random_state=seed)
    sample = sample.reset_index().rename(columns={"index": "test_index"})
    print(f"Test data source: {source}")
    print(f"Stratified sample: n = {len(sample)} (random_state = {seed}); per class: "
          + ", ".join(f"{LABEL_TO_NAME[k]}={v}" for k, v in
                      sorted(sample["label"].value_counts().items())))

    # ---- Part 1 model ---------------------------------------------------
    model, how = load_part1_model()
    print(f"Part 1 model: {how}")
    t0 = time.perf_counter()
    p1_pred = model.predict(sample["text"])
    p1_time = time.perf_counter() - t0

    # ---- Agent ----------------------------------------------------------
    print(f"\nAgent: LangChain create_agent, model '{CHAT_MODEL}' at {NRP_BASE_URL}, "
          f"temperature=0, reasoning_effort=low")
    agent = ClassificationAgent()
    rows, t_start = [], time.perf_counter()
    for i, r in sample.iterrows():
        out = agent.classify(r["text"])
        meta = out["_meta"]
        rows.append({
            "test_index": r["test_index"],
            "true": LABEL_TO_NAME[r["label"]],
            "part1_pred": LABEL_TO_NAME[int(p1_pred[i])],
            "agent_pred": out.get("category", "INVALID"),
            "agent_justification": out.get("justification", out.get("error")),
            "attempts": meta["attempts"],
            "latency_s": round(meta["latency_s"], 3),
            "input_tokens": meta["tokens"]["input"],
            "output_tokens": meta["tokens"]["output"],
            "reasoning_tokens": meta["tokens"]["reasoning"],
            "text": r["text"],
        })
        if (i + 1) % 25 == 0:
            done = pd.DataFrame(rows)
            print(f"  {i + 1:>3}/{len(sample)}  agent acc so far "
                  f"{(done['agent_pred'] == done['true']).mean():.3f}  "
                  f"avg latency {done['latency_s'].mean():.2f}s")
    agent_total = time.perf_counter() - t_start
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "part2_predictions.csv"), index=False)

    # ---- Metrics ----------------------------------------------------------
    y = df["true"]
    acc_p1 = accuracy_score(y, df["part1_pred"])
    acc_ag = accuracy_score(y, df["agent_pred"])
    prec_p1 = precision_score(y, df["part1_pred"], labels=CATEGORIES, average="macro", zero_division=0)
    prec_ag = precision_score(y, df["agent_pred"], labels=CATEGORIES, average="macro", zero_division=0)
    n_invalid = int((df["agent_pred"] == "INVALID").sum())
    n_retry = int((df["attempts"] > 1).sum())
    tok_in, tok_out = int(df["input_tokens"].sum()), int(df["output_tokens"].sum())
    tok_reason = int(df["reasoning_tokens"].sum())

    def cm_table(col):
        cm = confusion_matrix(y, df[col], labels=CATEGORIES)
        return pd.DataFrame(cm, index=[f"true {c}" for c in CATEGORIES],
                            columns=[f"pred {c}" for c in CATEGORIES]).to_string()

    print("\n" + "=" * 78 + "\nRESULTS\n" + "=" * 78)
    print(f"Sample size: {len(df)}")
    print(f"Part 1 MLP accuracy: {acc_p1:.4f}   macro precision: {prec_p1:.4f}")
    print(f"Agent accuracy:      {acc_ag:.4f}   macro precision: {prec_ag:.4f}")
    print(f"Agent invalid outputs after retry: {n_invalid}; examples needing a retry: {n_retry}")
    print(f"Agent latency: mean {df['latency_s'].mean():.2f}s, median {df['latency_s'].median():.2f}s, "
          f"max {df['latency_s'].max():.2f}s; total {agent_total:.1f}s")
    print(f"Part 1 inference: {p1_time:.3f}s total ({p1_time / len(df) * 1000:.2f} ms/example)")
    print(f"Agent tokens: input {tok_in:,}, output {tok_out:,} (of which reasoning {tok_reason:,})")
    print("\nPart 1 confusion matrix:\n" + cm_table("part1_pred"))
    print("\nAgent confusion matrix:\n" + cm_table("agent_pred"))

    both_wrong = df[(df["agent_pred"] != y) & (df["part1_pred"] != y)]
    agent_only = df[(df["agent_pred"] != y) & (df["part1_pred"] == y)]
    p1_only = df[(df["agent_pred"] == y) & (df["part1_pred"] != y)]
    print(f"\nDisagreement: agent-only errors {len(agent_only)}, Part1-only errors {len(p1_only)}, "
          f"both wrong {len(both_wrong)}")
    print("\nAgent errors with justifications:")
    for _, r in df[df["agent_pred"] != y].iterrows():
        print(f"\n  [test {r['test_index']}] true={r['true']} agent={r['agent_pred']} "
              f"part1={r['part1_pred']}\n  text: {r['text'][:220]}\n"
              f"  justification: {r['agent_justification']}")

    # ---- Comparison table ---------------------------------------------------
    table = pd.DataFrame({
        "Metric": [
            "Sample (stratified, seed 42)",
            "Accuracy",
            "Macro precision",
            "Labeled training examples used",
            "Avg latency per example",
            "Total evaluation time",
            "Invalid outputs (after 1 retry)",
            "Tokens (in / out)",
            "Cost / resources",
        ],
        "Part 1: TF-IDF + MLP": [
            f"{len(df)}",
            f"{acc_p1:.3f}",
            f"{prec_p1:.3f}",
            "120,000",
            f"{p1_time / len(df) * 1000:.2f} ms",
            f"{p1_time:.2f} s (+ ~65 s one-time training)",
            "n/a",
            "n/a",
            "Free; trains and runs on a laptop CPU (no GPU), ~8.3M parameters",
        ],
        f"Part 2: {CHAT_MODEL} agent (zero-shot)": [
            f"{len(df)}",
            f"{acc_ag:.3f}",
            f"{prec_ag:.3f}",
            "0 (instructions only)",
            f"{df['latency_s'].mean():.2f} s",
            f"{agent_total:.1f} s",
            f"{n_invalid} ({n_retry} needed retry)",
            f"{tok_in:,} / {tok_out:,}",
            "No price returned by the NRP Nautilus endpoint (no per-token billing for course access); cost reported as token usage. "
            "gpt-oss-120b (117B MoE, ~5.1B active) needs ~80 GB GPU memory",
        ],
    })
    with open(os.path.join(RESULTS_DIR, "part2_comparison.md"), "w", encoding="utf-8") as f:
        f.write(table.to_markdown(index=False) + "\n")
    print("\n" + table.to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Zero-shot AG News classification agent")
    ap.add_argument("article", nargs="?", help="article text (or pipe it on stdin)")
    ap.add_argument("--eval", action="store_true", help="compare with Part 1 on a test sample")
    ap.add_argument("--n", type=int, default=200, help="sample size for --eval (default 200)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt template")
    args = ap.parse_args()

    if args.show_prompt:
        print("SYSTEM:\n" + SYSTEM_PROMPT + "\n\nUSER:\n" + USER_TEMPLATE)
        return
    if args.eval:
        run_eval(args.n, args.seed)
        return

    article = args.article or (sys.stdin.read() if not sys.stdin.isatty() else "")
    if not article.strip():
        ap.error("provide an article as an argument or on stdin")
    result = ClassificationAgent().classify(article)
    print(public_json(result))
    meta = result["_meta"]
    print(f"(attempts: {meta['attempts']}, latency: {meta['latency_s']:.2f}s)", file=sys.stderr)
    if "error" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
