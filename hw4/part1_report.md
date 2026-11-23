# Part 1: Text Data Preprocessing, Vectorization, and Classification

All numbers below come from `python topic_classifier.py` (full log: `results/part1_output.txt`).

## (1) Load, inspect, and vectorize the data

**(a) Loading.** The script tries `torchtext.datasets.AG_NEWS` first. torchtext is deprecated (its last release supports up to Python 3.12 and torch 2.3), so it cannot be installed in this environment (Python 3.13). As the assignment allows, the script falls back to the official AG News v3 train/test split stored locally as `data/train.jsonl` (120,000 rows) and `data/test.jsonl` (7,600 rows). These are the same 120k/7.6k examples torchtext would download.

**(b) Text field.** Each article's text is `title + ". " + description`. Light cleanup removes dataset artifacts: `\` escaped newlines become spaces, and HTML entities such as `#39;` and `quot;` are decoded to `'` and `"`.

**(c) Example articles and class counts.**

| Category | Example (first training article of the class) |
|---|---|
| 1: World | Venezuelans Vote Early in Referendum on Chavez Rule (Reuters). Reuters - Venezuelans turned out early and in large numbers on Sunday to vote in a historic referendum that will either remove left-wing President Hugo Chavez from office or give him a new mandate… |
| 2: Sports | Phelps, Thorpe Advance in 200 Freestyle (AP). AP - Michael Phelps took care of qualifying for the Olympic 200-meter freestyle semifinals Sunday… |
| 3: Business | Wall St. Bears Claw Back Into the Black (Reuters). Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again. |
| 4: Sci/Tech | 'Madden,' 'ESPN' Football Score in Different Ways (Reuters). Reuters - Was absenteeism a little high on Tuesday among the guys at the office? EA Sports would like to think it was because "Madden NFL 2005" came out that day… |

| Category | Train | Test |
|---|---:|---:|
| World | 30,000 | 1,900 |
| Sports | 30,000 | 1,900 |
| Business | 30,000 | 1,900 |
| Sci/Tech | 30,000 | 1,900 |
| **Total** | **120,000** | **7,600** |

Both splits are perfectly balanced, so accuracy is not inflated by a majority class, and macro and weighted averages are almost identical.

**(d) Vectorizers.** `CountVectorizer()` builds the DTM and `TfidfVectorizer()` builds the TF-IDF matrix. Both use lowercasing and the default token pattern (words of two or more characters). Each vectorizer is fit with `fit_transform` on the **training text only**, and the test text is mapped with `transform`. Test-only words are therefore ignored and no vocabulary or IDF information leaks from the test set.

**(e) Shape and sparsity of the training DTM.**

| Quantity | Value |
|---|---:|
| Shape (documents × features) | **120,000 × 64,889** |
| Nonzero entries | 3,665,376 |
| Total entries | 7,786,680,000 |
| Sparsity = (1 − 3,665,376 / 7,786,680,000) × 100 | **99.9529 %** |
| Avg. distinct terms per document | 30.5 |
| Dense float64 storage | 62.3 GB |
| CSR sparse storage | 44.5 MB |

**(f) Concepts.**

- A **document** is one unit of text being analyzed, here one news article (title + description). The **corpus** is the full collection of documents, here the 120,000 training articles. Only the training corpus defines the vocabulary.
- **DTM row** = one document. **DTM column** = one vocabulary term. **DTM cell (i, j)** = the raw number of times term *j* appears in document *i*.
- A **TF-IDF cell** reweights that count: tf(i, j) × idf(j), where idf(j) = ln((1+N)/(1+df(j))) + 1. Scikit-learn then L2-normalizes each row. Words that appear in many documents get down-weighted, and rare, informative words get up-weighted. For example, in training document 0 the words "of" and "cynics" each have count 1, but their TF-IDF values are 0.050 and 0.339. "Wall" (count 2) gets 0.345, but "reuters" (also count 2) gets only 0.200 because it appears in thousands of articles. TF-IDF cells are therefore real-valued, length-normalized weights rather than integer counts.
- **Why sparse:** the vocabulary has about 65k terms, but a short news blurb uses only about 30 distinct words. More than 99.95 % of each row is therefore zero.
- **Effect on storage and computation:** stored densely, the training DTM would need about 62 GB. Compressed Sparse Row (CSR) format stores only the nonzero values and their column indices (about 45 MB, roughly 1,400× smaller). Matrix products such as the MLP's first layer (X · W) also cost time proportional to the number of nonzeros, not rows × columns, so training on 120k × 65k is feasible on a laptop. The drawbacks are that most feature pairs never co-occur, and the very high dimensionality makes the first-layer weight matrix large (65k × 128 ≈ 8.3M parameters) and prone to overfitting rare words.

## (2) MLP classifier on CountVectorizer features

**Reproducible configuration** (`sklearn.neural_network.MLPClassifier`):

| Setting | Value |
|---|---|
| Input | 64,889 bag-of-words count features (CountVectorizer, no stop-word removal) |
| Architecture | 64,889 → **128** (hidden, ReLU) → 4 (softmax output) |
| Activation | ReLU (hidden), softmax (output) |
| Loss | Categorical cross-entropy (log-loss) + L2 penalty α = 1e-4 |
| Optimizer | Adam, β₁ = 0.9, β₂ = 0.999, ε = 1e-8 |
| Learning rate | 0.001, constant |
| Batch size | 256 |
| Epochs | 5 (`max_iter=5`, `early_stopping=False`, `shuffle=True`) |
| Random seed | `random_state=42` |
| Training time | ~63 s on an Apple M-series CPU (10 cores, 24 GB RAM) |

**Test-set results (7,600 held-out articles)**

| Metric | Value |
|---|---:|
| Accuracy | **0.9095** |
| Macro-averaged precision | **0.9097** |

*Averaging method:* macro precision is the unweighted mean of the four per-class precisions (World 0.9248, Sports 0.9587, Business 0.8916, Sci/Tech 0.8637). Each class counts equally regardless of size.

**Confusion matrix** (rows = true, columns = predicted). Image: `results/confusion_count_nostop.png`.

| | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| **World** | **1709** | 53 | 64 | 74 |
| **Sports** | 24 | **1857** | 11 | 8 |
| **Business** | 68 | 14 | **1629** | 189 |
| **Sci/Tech** | 47 | 13 | 123 | **1717** |

Sports is the easiest class (97.7 % recall) because its vocabulary (team names, scores, "season", "coach") is distinctive. The largest error is **Business ↔ Sci/Tech** (189 + 123 = 312 of the 688 errors, 45 %). Tech-company news (earnings of Intel, Google IPO, Oracle–PeopleSoft) uses both financial and technology vocabulary.

**Three misclassified test examples**

| # | Text (abridged) | True | Predicted | Plausible explanation |
|---|---|---|---|---|
| 1 | "E-mail scam targets police chief. Wiltshire Police warns about 'phishing' after its fraud squad chief was targeted." | Sci/Tech | World | Most tokens are crime/law-enforcement words ("police", "chief", "warns", "fraud squad", a UK place name) that appear mostly in World news. The one tech cue, "phishing", is rare in the training data, so it carries little weight against the many World-associated words. |
| 2 | "Haas and Cink Selected for Ryder Cup Team. Jay Haas joined Stewart Cink as the two captain's picks for a U.S. team that will try to regain the cup from Europe next month." | Sports | World | There are no score or game words. "U.S.", "Europe", "selected", and "regain" read like international affairs. The player names are rare tokens, and bag-of-words cannot tell that "Ryder Cup" is a golf event. |
| 3 | "Oil prices bubble to record high. The price of oil … closing at a record high. The main contract in New York, light sweet crude … $US46.75 a barrel – up 70 cents…" | World | Business | This is arguably label noise or genuine ambiguity. The text is entirely market language ("prices", "contract", "barrel", "close", "$"), which is exactly what Business articles look like. Oil stories in AG News are split between World and Business depending on the news source. |

**Why documents, not terms, are the observations.** In supervised document classification the label (World/Sports/Business/Sci/Tech) belongs to each *article*. Each training example must therefore be one document (a DTM row) with its label, and the terms (columns) are the features that describe it. The model learns P(category | words in the document) and is applied to new documents at test time. If you **transpose** the DTM, each row becomes a *term* described by its counts across the documents. There is no category label for a term, so the target would have to change to something like "which topic does this word belong to". That is a different task (term classification or word clustering). The features would also be tied to specific training documents, so the model could not be applied to a new, unseen article. Transposing does not give an equivalent view of the same problem. It changes what is being predicted.

## (3) Feature representation and stop-word removal

Same split, same MLP (128 ReLU units, Adam, lr 0.001, batch 256, 5 epochs, seed 42). Each vectorizer is fit on training text only. Stop-word removal uses scikit-learn's built-in English list (318 words).

| Vectorizer | Stop-word removal | # features | Test accuracy | Macro precision |
|---|---|---:|---:|---:|
| CountVectorizer | No | 64,889 | 0.9095 | 0.9097 |
| **TfidfVectorizer** | **No** | 64,889 | **0.9113** | **0.9113** |
| CountVectorizer | Yes | 64,585 | 0.9088 | 0.9090 |
| TfidfVectorizer | Yes | 64,585 | 0.9105 | 0.9108 |

**Confusion matrix of the best configuration (TF-IDF, no stop-word removal).** Image: `results/confusion_tfidf_nostop.png`.

| | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| **World** | **1708** | 55 | 76 | 61 |
| **Sports** | 24 | **1859** | 10 | 7 |
| **Business** | 66 | 13 | **1647** | 174 |
| **Sci/Tech** | 46 | 13 | 129 | **1712** |

**Discussion.**

- **TF-IDF helped slightly** (+0.18 points with stop words kept, +0.17 with them removed). It gave the best result in both stop-word settings. TF-IDF down-weights words that appear everywhere and L2-normalizes each document. All inputs then have similar scale regardless of article length, which suits gradient-based MLP training. Its gain is mostly fewer Business→Sci/Tech errors (189 → 174).
- **Stop-word removal made essentially no difference, and was slightly worse** (−0.07 to −0.08 points). It only removes about 300 of 65k features. Those words carry little topic information, and the MLP (and TF-IDF's IDF term) already learn to give them near-zero weight. Removing them also discards a few useful cues, for example "us"/"we" in political quotes.
- **These differences are within noise.** With 7,600 test examples, the standard error of an accuracy near 0.91 is √(0.91·0.09/7600) ≈ 0.33 points. All four configurations fall within 0.25 points of each other. The safe conclusion is that for topic classification with a well-regularized MLP on 120k examples, the choice of weighting and stop-word handling matters far less than the bag-of-words representation itself. The remaining errors come from genuinely overlapping topics (Business vs. Sci/Tech, World vs. Business) that no word-weighting scheme can fully resolve.

## (4) Sentiment-analysis pipeline for Business news (~100 words)

**Data:** collect Business articles (e.g., AG News class 3 plus financial-news headlines) and deduplicate. **Labels:** annotate each article as *positive / neutral / negative* from an investor's point of view, using at least two annotators and measuring agreement (Cohen's κ), or use a labeled set such as Financial PhraseBank. **Features:** TF-IDF unigrams + bigrams *without* removing negations ("not", "no"), plus a finance-specific lexicon (Loughran–McDonald), or fine-tuned FinBERT embeddings. **Model and evaluation:** stratified train/validation/test split; report accuracy, macro-F1, and a confusion matrix, since neutral usually dominates. **Difference from topic classification:** topic depends on *what* an article is about, so content words dominate and stop words or negation barely matter. Sentiment depends on *how* the event is described ("profits fell" vs. "profits rose"), so word order, negation, and domain polarity ("volatile", "beat estimates") are essential, and labels are more subjective.
