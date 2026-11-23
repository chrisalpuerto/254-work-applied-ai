"""
CPSC 254 - Assignment 4, Part 3 (2)
Evaluate the syllabus RAG application using the SAVED Chroma index.

Run `python rag.py --build` once first, then:
    python rag_eval.py

For every question this prints the generated answer, the retrieved chunks (page and
cosine distance), the reference answer verified by hand from the syllabus, and an
automatic key-fact check. The 0/1/2 accuracy/completeness/relevance ratings are
assigned by a human reviewer in the report, using this output.

Outputs: results/part3_eval_output.txt, results/part3_eval.json
"""

import json
import os
import re
import sys
import time
import unicodedata

from rag import NOT_ENOUGH_INFO, SyllabusRAG

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

# Each "key_facts" entry is a list of acceptable spellings (case-insensitive);
# a fact counts as present if any spelling appears in the answer.
EVAL_SET = [
    {
        "id": "Q1",
        "question": "How is the final grade weighted between class participation and "
                    "attendance, assignments, and exams?",
        "reference": "Class participation and attendance 7%, Assignments 43%, Exams 50% "
                     "(total 100%); final grade is the weighted sum.",
        "evidence": "p. 3, 'Grading Policy' table",
        "key_facts": [["7%", "7 %"], ["43%", "43 %"], ["50%", "50 %"]],
    },
    {
        "id": "Q2",
        "question": "What are the instructor's office hours and where is the office?",
        "reference": "Mon and Wed 2:15-4:00 pm and 5:15-5:30 pm, or by Zoom appointment; "
                     "office CS-421.",
        "evidence": "p. 1, 'Instructor' / 'Office Hours'",
        "key_facts": [["CS-421", "CS 421"], ["2:15"], ["4:00"], ["5:15"], ["5:30"], ["zoom"]],
    },
    {
        "id": "Q3",
        "question": "What happens if I miss an exam with a valid, documented excuse, "
                    "and what if I have no valid excuse?",
        "reference": "With a valid documented excuse (e.g., doctor's note) the weight of the "
                     "final is increased by the weight of the missed exam; without one the "
                     "exam score is zero and the final weight is unchanged.",
        "evidence": "p. 2, 'Exams' - Makeup policy",
        "key_facts": [["final"], ["weight"], ["zero", " 0 "]],
    },
    {
        "id": "Q4",
        "question": "What letter grade does a final score of 90% earn, and what is the "
                    "minimum score for an A?",
        "reference": "90% is the minimum for an A- (A- >= 90%); an A requires >= 92%.",
        "evidence": "p. 3, 'Grading Policy' - plus/minus scheme",
        "key_facts": [["A-"], ["92"]],
    },
    {
        "id": "Q5",
        "question": "Am I allowed to use AI tools such as large language models for "
                    "assignments and during exams?",
        "reference": "AI tools are permitted/encouraged for assignments (e.g., writing code, "
                     "rephrasing short answers) but you must understand submitted AI code; no "
                     "AI tools during exams unless otherwise stated.",
        "evidence": "p. 3, 'AI Use Policy'",
        "key_facts": [["permitted", "encouraged", "allowed"], ["exam"], ["understand"]],
    },
    {
        "id": "Q6 (not in syllabus)",
        "question": "What is the due date for Assignment 4?",
        "reference": "Not in the syllabus. It only says assignment details will be announced "
                     "later. Expected behaviour: the NOT_ENOUGH_INFO response.",
        "evidence": "p. 2, 'Assignments' (no dates given)",
        "key_facts": [[NOT_ENOUGH_INFO.lower()]],
    },
]


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def normalize(text):
    """Undo typographic variants the LLM uses (narrow no-break spaces, Unicode
    hyphens, markdown bold) so the key-fact check compares plain text."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    return text.replace("**", "").replace(" ", " ")


def key_fact_check(answer, facts):
    a = f" {normalize(answer).lower()} "
    found = [any(alt.lower() in a for alt in alts) for alts in facts]
    return sum(found), len(found), [alts[0] for alts, ok in zip(facts, found) if not ok]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(RESULTS_DIR, "part3_eval_output.txt"))

    print("=" * 78)
    print("CPSC 254 Syllabus RAG - evaluation run")
    print("=" * 78)
    rag = SyllabusRAG(require_existing=True)  # loads the saved index, never rebuilds
    cfg = rag.config
    print(f"Chat model: {cfg['chat_model']} | embeddings: {cfg['embedding_model']} | "
          f"chunk_size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}, "
          f"chunks={cfg['n_chunks']}, top_k={rag.k}")

    records, t0 = [], time.perf_counter()
    for item in EVAL_SET:
        res = rag.ask(item["question"])
        hit, total, missing = key_fact_check(res["answer"], item["key_facts"])
        print("\n" + "-" * 78)
        print(f"{item['id']}: {item['question']}")
        print("-" * 78)
        print(f"ANSWER:\n{res['answer']}\n")
        print("RETRIEVED: " + ", ".join(f"p.{s['pages']} (chunk {s['chunk_id']}, "
                                         f"dist {s['distance']:.3f})" for s in res["sources"]))
        print(f"REFERENCE ({item['evidence']}): {item['reference']}")
        print(f"KEY FACTS PRESENT: {hit}/{total}"
              + (f"  missing: {missing}" if missing else "")
              + f"   | insufficient-info flag: {res['insufficient_info']}"
              + f"   | latency {res['latency_s']:.1f}s")
        records.append({**item, **res, "key_facts_found": f"{hit}/{total}"})

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 78)
    print(f"Evaluated {len(records)} questions in {elapsed:.1f}s "
          f"(avg {elapsed / len(records):.1f}s per question)")
    for r in records:
        print(f"  {r['id']:<22} key facts {r['key_facts_found']:<5} "
              f"pages retrieved: {', '.join(s['pages'] for s in r['sources'])}")

    with open(os.path.join(RESULTS_DIR, "part3_eval.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {RESULTS_DIR}/part3_eval.json and part3_eval_output.txt")


if __name__ == "__main__":
    main()
