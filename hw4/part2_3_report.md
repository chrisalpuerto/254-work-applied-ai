# Parts 2 and 3: LLM Classification Agent and Syllabus RAG

**Environment:** Python 3.13, LangChain 1.5 (`create_agent`), langchain-openai 1.8, langchain-chroma 1.2 / chromadb 1.5. The LLM is `gpt-oss` on the NRP Nautilus LLM service, an OpenAI-compatible endpoint that serves `openai/gpt-oss-120b`. The API key is read from `.env` and is not included in the submission.

# Part 2: LLM-based Classification Agent

## (1) Agent implementation (`classification_agent.py`)

**(a) Agent.** The `ClassificationAgent` class uses LangChain's `create_agent(model, tools=[], system_prompt=…)`. The model is `ChatOpenAI(model="gpt-oss", temperature=0, reasoning_effort="low")`, pointed at the Nautilus endpoint (settings in `nrp_llm.py`). No tools are needed because the task is a single read-and-label step. `classify(article)` returns the result for one AG News article:

```
$ python classification_agent.py "Haas and Cink Selected for Ryder Cup Team. Jay Haas joined Stewart Cink as the two captain's picks for a U.S. team that will try to regain the cup from Europe next month."
{"category": "Sports", "justification": "Mentions Ryder Cup team selections, a major golf competition."}
(attempts: 1, latency: 2.94s)
```

The Part 1 MLP misclassified this same article as World.

**(b) Zero-shot prompt template** (printed by `python classification_agent.py --show-prompt`). It contains category definitions but **no labeled examples**:

```
SYSTEM:
You are a news topic classification agent for the AG News dataset.
Classify the news article given by the user into exactly ONE of these four categories:

- World: international and national affairs, politics, government, elections, war and conflict, diplomacy, crime, disasters
- Sports: sports events, athletes, teams, games, leagues, tournaments
- Business: companies, markets, the economy, finance, earnings, trade, industry, labor
- Sci/Tech: science, technology, computing, the internet, software, space, scientific and medical research

Respond with ONLY a single JSON object and nothing else (no markdown, no code fences),
with exactly these two keys:
{"category": "<one of: World, Sports, Business, Sci/Tech>", "justification": "<one short sentence, at most 30 words, naming the key cues in the article>"}

Do not include step-by-step reasoning. The justification must be a concise explanation only.

USER:
Article:
{article}
```

**(c) Output format.** The output is valid JSON with exactly `category` and `justification`, defined as a Pydantic model:

```python
class TopicClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")                 # no other keys allowed
    category: Literal["World", "Sports", "Business", "Sci/Tech"]
    justification: str                                        # non-empty, <= 40 words
```

**(d) Validation and error handling.** `parse_and_validate()` checks the reply in this order:

1. It strips an optional code fence.
2. It runs `json.loads`, which catches non-JSON output.
3. It checks that the result is a JSON object.
4. It normalizes harmless case and spacing in the category (e.g., `"sci/tech"` → `"Sci/Tech"`).
5. It validates against the schema: allowed category, no extra keys, and a non-empty, brief justification.

If validation fails, the agent **retries once**. It sends back its invalid reply plus the specific error message and asks for JSON only. If the second reply also fails, `classify` returns a clear error object, `{"error": "Validation failed after retry: …", "raw_output": "…"}`, instead of crashing.

Tested inputs:

| Input | Result |
|---|---|
| `not json` | rejected: *output is not valid JSON* |
| `{"category":"Politics",…}` | rejected: *category: Input should be 'World', 'Sports', 'Business' or 'Sci/Tech'* |
| an extra `"confidence"` key | rejected: *Extra inputs are not permitted* |

**No hidden chain-of-thought:** gpt-oss is a reasoning model, but its reasoning tokens are returned separately by the server and are **never read or shown**. `reasoning_effort="low"` keeps them short (about 13 tokens per example). Only the one-sentence justification in the JSON is exposed.

## (2) Comparison with the best Part 1 classifier

**Setup:**

- **Sample:** a stratified random sample of **n = 200** from the official 7,600-article test set (`train_test_split(…, train_size=200, stratify=label, random_state=42)`), 50 per class. The full 200 were evaluated, so no reduction was needed.
- **Part 1 model:** the best Part 1 configuration (TF-IDF without stop-word removal + MLP 128 ReLU, Adam, 5 epochs, seed 42). It was trained on the 120,000 training articles only. The 200 test examples were not used for training or tuning.
- **Agent:** called sequentially, one article at a time, so the latencies are true per-request times.
- **Outputs:** per-example predictions in `results/part2_predictions.csv`, full log in `results/part2_output.txt`.

**Comparison table**

| | **Part 1: TF-IDF + MLP** | **Part 2: gpt-oss zero-shot agent** |
|---|---|---|
| Sample (same examples) | 200 (stratified, seed 42) | 200 (stratified, seed 42) |
| **Accuracy** | **0.870** (174/200) | 0.855 (171/200) |
| Macro precision | 0.870 | 0.869 |
| Labeled training data | 120,000 articles | **0** (task description only) |
| **Latency per example** | 0.03 ms | **2.25 s** mean (median 2.03 s, max 8.53 s) |
| **Total evaluation time** | 0.007 s (+ ~65 s one-time training) | **449.9 s** (~7.5 min) |
| Invalid outputs / retries | n/a | 0 invalid, 0 retries (200/200 valid JSON on first try) |
| Tokens | n/a | 64,084 input + 10,555 output (2,623 of them reasoning); ~320 in / 53 out per article |
| **Cost / resources** | Free. Trains in ~65 s and predicts on a laptop CPU, no GPU; ~8.3M parameters. | No price is returned by the endpoint and no per-token billing applies to the course's Nautilus access, so cost is reported as token usage. At typical commercial gpt-oss-120b API prices (~$0.10–0.15 per 1M input tokens, ~$0.50–0.60 per 1M output tokens) this run would cost **under $0.02**. Model: gpt-oss-120b, a 117B-parameter mixture-of-experts with ~5.1B active per token, which needs ~80 GB of GPU memory (one H100/A100-80GB) to self-host. |
| **Data efficiency** | Needs a large labeled corpus. Its accuracy comes entirely from 120k labeled examples, and a new label set means relabeling and retraining. | Gets within 1.5 points of the trained model with **zero** labeled examples, using only a four-line category description. Changing categories means editing the prompt. A few-shot prompt could add a handful of examples of the dataset's conventions (e.g., tech-company news → Sci/Tech). |
| **Explanations / error analysis** | No per-prediction explanation. Errors can only be analysed after the fact from the confusion matrix and by reading misclassified texts (as in Part 1). | Every prediction has a readable justification naming the cues used. This makes it easy to see *why* it chose a label and to spot label noise. However, the justification is a post-hoc rationale generated by the model, not a faithful trace of its computation, and it sounds equally confident when wrong. |
| **Main limitations** | Bag-of-words: no word order or world knowledge, and it fails when the topic words are rare or misleading. Fixed to AG News' four labels. | About 75,000× slower, depends on the network and a remote GPU service (availability, rate limits), and is less reproducible if the served model changes. It does not know AG News' labeling conventions, so its Sci/Tech recall is low. |

**Confusion matrices** (rows = true, columns = predicted)

| Part 1 MLP | World | Sports | Business | Sci/Tech |  | Agent | World | Sports | Business | Sci/Tech |
|---|--:|--:|--:|--:|---|---|--:|--:|--:|--:|
| **World** | 43 | 2 | 4 | 1 | | **World** | 44 | 1 | 5 | 0 |
| **Sports** | 1 | 47 | 2 | 0 | | **Sports** | 1 | 48 | 1 | 0 |
| **Business** | 1 | 1 | 40 | 8 | | **Business** | 0 | 1 | **46** | 3 |
| **Sci/Tech** | 3 | 1 | 2 | 44 | | **Sci/Tech** | 5 | 0 | **12** | 33 |

**Discussion.**

- **Accuracy on this sample.** Both methods score lower here than Part 1's full-test 0.911 (the MLP gets 0.870 on these 200), so this sample happens to be harder than average. With n = 200 the 1.5-point gap (3 articles) is well within sampling noise (standard error ≈ 2.4 points).
- **The two models fail differently.** The agent beats the MLP on World, Sports, and especially Business (46 vs. 40 correct). It loses badly on Sci/Tech (33 vs. 44): **12 Sci/Tech articles were labeled Business**, for example "Dell Debuts Color Laser Printers", "Cray promotes two execs", "Gartner: Q2 server shipments rise", and Apple iTunes pricing stories. The agent's justifications show it is applying a reasonable human reading ("corporate executive promotions … a business news item"). AG News instead files most tech-*company* news under Sci/Tech, a convention the MLP learned from 120k examples and the zero-shot agent cannot know.
- **Overlap of errors.** The agent made 14 errors the MLP did not, the MLP made 11 the agent did not, and **15 articles fooled both**. Many of the shared errors look like label noise or genuinely ambiguous articles:
  - "Cricket: Tendulkar to miss test" is labeled **World**.
  - "IBM Claims Its BlueGene Supercomputer Is the Fastest" is labeled **Business**.
  - "Tyncastle sale gets official go ahead" (the sale of a football club's stadium, written as a shareholder-vote story) is labeled **Sports**.
  - A journalist's biography ("Hayley Mick") is labeled **World**.

  The agent's justifications for these ("Mentions cricket and a test match involving Sachin Tendulkar, a sports event") make the label problem obvious, which the MLP's bare predictions never would.
- **Justifications vs. traditional error analysis.** The agent's justifications are useful for quick auditing, debugging the prompt, and finding mislabeled data. However, they are generated after the decision, cannot be verified against the model's internals, and look equally plausible when wrong (e.g., "Supreme Court hearing on file-sharing law involves government and legal policy" for an article labeled Sci/Tech). The MLP's error analysis is aggregate and quantitative (confusion matrix, per-class precision/recall), reproducible, and reflects exactly what the model does, but it needs a human to read the errors to understand *why*.
- **Practical takeaway.** With plenty of labeled data and a fixed label set, the trained classifier is as accurate, about 75,000× faster, and essentially free at inference. The zero-shot agent is the better choice when labeled data is scarce, categories change often, or explanations are needed. Adding a few examples of the dataset's conventions to the prompt (few-shot) is the obvious next step to fix its Sci/Tech-vs-Business bias.

---

# Part 3: Retrieval-Augmented Generation over the Course Syllabus

## (1) RAG pipeline (`rag.py`)

| Step | Implementation |
|---|---|
| (a) Load PDF | `pypdf.PdfReader` extracts the text of the 5 pages of `CPSC 254_SYL.pdf` (11,688 characters after cleaning). The header repeated on every page ("Syllabus … Applied Artificial Intelligence (CPSC 254)") and the "1/5" page counters are removed so they don't pollute every chunk. |
| (b) Chunking | `RecursiveCharacterTextSplitter`, **chunk_size = 800 characters, chunk_overlap = 150 characters**. Separators are tried in order: paragraph → line → sentence → word. The whole document is split as one string, so a policy that continues onto the next page (e.g., the drop policy, pp. 1–2) stays in one chunk. Each chunk records its start offset, which is mapped back to page numbers. Result: **22 chunks**. |
| (c) Embeddings and vector DB | `qwen3-embedding` on Nautilus (4,096-dimensional vectors) via `OpenAIEmbeddings`. Vectors, chunk text, and metadata (`chunk_id`, `page_start`, `page_end`, `pages`) are stored in a persistent **Chroma** collection `cpsc254_syllabus` using cosine distance. |
| (d) Retrieve and generate | `similarity_search_with_score` returns the **top k = 4** chunks. They are inserted into the prompt as `[Excerpt i \| page p]` blocks together with the question and sent to `gpt-oss` (temperature 0). |
| (e) Persistence | The index is saved in `chroma_syllabus/` (`chroma.sqlite3` + HNSW files) and `chroma_syllabus/rag_config.json` records the PDF's SHA-256, chunk size/overlap, splitter, embedding model, collection name, distance metric, top-k, chunk count, and build time. On startup `rag.py` compares this config with the current settings and PDF hash. The index is **reused** if everything matches and the chunk count is correct; otherwise it is rebuilt. `--rebuild` forces a rebuild. `rag_eval.py` uses `require_existing=True`, so it never rebuilds. |
| (f) Insufficient info | The system prompt requires answers from the excerpts only, with page citations and no guessed dates, numbers, or policies. If the excerpts do not contain the answer, the model must reply exactly *"The syllabus does not contain enough information to answer this question."* The code flags such answers (`insufficient_info = True`). If only part of a question is covered, the model must say which part is missing. |

**Prompt used for generation**

```
SYSTEM:
You answer questions about the CPSC 254 (Applied Artificial Intelligence) course
syllabus. Use ONLY the syllabus excerpts provided in the user message.

Rules:
- Base every statement on the excerpts. Do not use outside knowledge and do not guess
  dates, numbers, names, or policies that are not written in the excerpts.
- Cite the page(s) you used in square brackets, e.g. [p. 3].
- If the excerpts do not contain the answer, reply exactly:
  "The syllabus does not contain enough information to answer this question."
  You may add one sentence saying what related information the syllabus does give.
- If only part of the question is answered by the excerpts, answer that part and state
  clearly which part the syllabus does not cover.
- Be concise: at most 5 sentences or a short bullet list, in plain text (no bold or
  other markdown formatting).

USER:
Syllabus excerpts:
[Excerpt 1 | page 3]
<chunk text>
...
Question: <question>
```

**Why 800/150?** Syllabus sections (Grading Policy, AI Use Policy, Makeup policy) are typically 300–900 characters. An 800-character chunk usually holds one complete section, so the retrieved text contains the whole rule rather than half of it. The 150-character (~19 %) overlap keeps sentences that straddle a boundary intact in at least one chunk. With only 22 chunks, k = 4 retrieves about 18 % of the syllabus, which is enough context without drowning the answer in unrelated policies.

## (2) Evaluation (`rag_eval.py`)

`rag_eval.py` loads the saved index (it exits with an error if no valid index exists) and asks six questions. Q1–Q5 can be checked directly against the syllabus. Q6 is deliberately unanswerable, to test requirement (f). For each question the script prints the answer, the retrieved chunks with page numbers and cosine distances, the hand-verified reference answer, and an automatic key-fact check. Full log: `results/part3_eval_output.txt`. All 6 questions ran in 32 s (5.4 s per question on average).

Ratings: **Acc** = accuracy, **Comp** = completeness, **Rel** = relevance. Each is rated 0 = poor, 1 = partly correct, 2 = good.

| # | Question | Generated answer (verbatim, citations trimmed) | Supporting syllabus text | Acc | Comp | Rel | Justification and issues |
|---|---|---|---|:-:|:-:|:-:|---|
| Q1 | How is the final grade weighted between class participation and attendance, assignments, and exams? | "Class participation and attendance: 7% · Assignments: 43% · Exams: 50% [page 3]" | p. 3, *Grading Policy* table: "Class participation and attendance 7% / Assignments 43% / Exams 50% / Total 100%" | 2 | 2 | 2 | All three weights are exact and the page is correct. The top retrieved chunk (distance 0.157) was the grading table. No unsupported claims. |
| Q2 | What are the instructor's office hours and where is the office? | "Mon and Wed 2:15 pm – 4:00 pm and 5:15 pm – 5:30 pm, or by ZOOM appointment. Office location: CS-421. [page 1]" | p. 1: "Office: CS-421 … Office Hours: Mon and Wed 2:15 pm – 4:00 pm and 5:15pm – 5:30 pm or ZOOM appointment" | 2 | 2 | 2 | Both time windows, the Zoom option, and the room are correct (6/6 key facts). No omissions. |
| Q3 *(specific policy)* | What happens if I miss an exam with a valid, documented excuse, and what if I have no valid excuse? | "With a valid, documented excuse, the exam's weight is added to the final exam weight … Without a valid excuse, you receive a zero on the exam/quiz and the final exam weight does not change. [p. 2-3]" | p. 2, *Exams – Makeup policy*: "…the weight of your final will be increased by the weight of the exam. Without a valid excuse, you will receive a zero score for the exam/quiz, and the final weight will remain unchanged." | 2 | 1 | 2 | It faithfully restates both cases. **Omission:** the same page also says "Final exam: No final exam". The syllabus contradicts itself, and the answer doesn't point this out. It also leaves out the examples of a valid excuse (doctor's note, death in the family). |
| Q4 *(specific detail)* | What letter grade does a final score of 90% earn, and what is the minimum score for an A? | "A final score of 90 % earns an A-. The minimum score required for an A is 92 %. [p. 3]" | p. 3, plus/minus scheme: "A: >= 92%   A-: >= 90%" | 2 | 2 | 2 | Both thresholds are correct, including the boundary case where 90 % counts as A- because the rule is ≥ 90 %. |
| Q5 *(specific policy)* | Am I allowed to use AI tools such as large language models for assignments and during exams? | "Yes, you may use AI tools (including LLMs) for completing assignments or projects, provided you understand the generated code and can show your contribution. No, AI tools are not allowed during exams unless a specific exception is announced, as exams are closed-book and closed-notes. [p. 3]" | p. 3, *AI Use Policy*: "…permitted or encouraged for completing assignments/projects … you should understand any AI-generated code you submit and be able to demonstrate your contribution … No AI tools are allowed during exams unless otherwise stated" | 2 | 2 | 2 | Both parts are correct. **Minor citation error:** "closed-book and closed-notes" is true, but it comes from the *Exams* section on **p. 2** (retrieved as Excerpt 2), not p. 3 as cited. It omits the example uses ("writing code and rephrasing short answers"). |
| Q6 *(not in syllabus)* | What is the due date for Assignment 4? | "The syllabus does not contain enough information to answer this question. It does note that important dates such as campus closures, holidays, and the final exam schedule are available on the university's Academic Calendar and the Final Exam Schedule page [p. 1]." | Nothing. p. 2, *Assignments*, says only "Detailed information about assignments/projects will be announced later." | 2 | 1 | 2 | It correctly refuses and invents no date, which is the intended behaviour (f). The extra sentence is supported by p. 1. **Omission:** the most relevant related fact ("details … will be announced later", p. 2) was not retrieved. All four distances were high (0.52–0.55), meaning no chunk matched well. |

**Overall:** 5 of the 5 answerable questions were factually correct, and the unanswerable question was refused rather than hallucinated. Across the six questions the answers scored 12/12 for accuracy, 10/12 for completeness, and 12/12 for relevance. The weak points are (1) the model reports what the retrieved text says without spotting internal contradictions in the source (Q3), (2) page citations are occasionally attached to the wrong excerpt (Q5), and (3) when retrieval misses the most relevant chunk, the "related information" sentence is less helpful (Q6). Retrieval distance is a useful signal: answerable questions had a best-chunk distance of 0.10–0.38, while the unanswerable one was above 0.5.

**Evidence of a successful run:** see `results/part3_eval_output.txt`, or take a terminal screenshot of `python rag_eval.py` for the report.
