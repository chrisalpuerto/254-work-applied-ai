"""
CPSC 254 - Assignment 4, Part 3
Retrieval-Augmented Generation over the course syllabus ("CPSC 254_SYL.pdf")
with a persistent Chroma vector database and gpt-oss on NRP Nautilus.

Usage:
    python rag.py --build                      # build (or verify) the index only
    python rag.py "What are the grading weights?"
    python rag.py                              # interactive Q&A loop
    python rag.py --rebuild "question"         # force re-indexing

Pipeline:
  (a) load the PDF and extract text page by page (pypdf), dropping the repeated header
  (b) split into overlapping chunks: CHUNK_SIZE=800 characters, CHUNK_OVERLAP=150
  (c) embed chunks with 'qwen3-embedding' and store vectors + source text + page
      metadata in a persistent Chroma collection
  (d) retrieve the top-k (k=4) chunks for a question and ask gpt-oss to answer
      ONLY from those chunks, citing pages
  (e) the index and rag_config.json are saved in ./chroma_syllabus/ and reused on
      later runs; the index is rebuilt only if the PDF or the settings change
  (f) if the retrieved syllabus text is insufficient, the model must reply with
      NOT_ENOUGH_INFO instead of inventing details
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

from nrp_llm import CHAT_MODEL, EMBED_MODEL, get_chat_model, get_embeddings

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Configuration (saved to rag_config.json next to the index)
# ---------------------------------------------------------------------------
PDF_PATH = os.path.join(HERE, "CPSC 254_SYL.pdf")
PERSIST_DIR = os.path.join(HERE, "chroma_syllabus")
CONFIG_PATH = os.path.join(PERSIST_DIR, "rag_config.json")
COLLECTION = "cpsc254_syllabus"

CHUNK_SIZE = 800      # characters per chunk (~150-200 tokens, about one syllabus section)
CHUNK_OVERLAP = 150   # characters shared by consecutive chunks so sentences on a boundary are kept
TOP_K = 4             # chunks retrieved per question

NOT_ENOUGH_INFO = "The syllabus does not contain enough information to answer this question."

SYSTEM_PROMPT = f"""You answer questions about the CPSC 254 (Applied Artificial Intelligence) course
syllabus. Use ONLY the syllabus excerpts provided in the user message.

Rules:
- Base every statement on the excerpts. Do not use outside knowledge and do not guess
  dates, numbers, names, or policies that are not written in the excerpts.
- Cite the page(s) you used in square brackets, e.g. [p. 3].
- If the excerpts do not contain the answer, reply exactly:
  "{NOT_ENOUGH_INFO}"
  You may add one sentence saying what related information the syllabus does give.
- If only part of the question is answered by the excerpts, answer that part and state
  clearly which part the syllabus does not cover.
- Be concise: at most 5 sentences or a short bullet list, in plain text (no bold or
  other markdown formatting)."""

USER_TEMPLATE = """Syllabus excerpts:
{context}

Question: {question}"""

_HEADER_RE = re.compile(r"^\s*Syllabus\s+Applied Artificial Intelligence \(CPSC 254\)\s*$", re.M)
_PAGENUM_RE = re.compile(r"^\s*\d+/\d+\s*$", re.M)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def current_settings():
    return {
        "pdf_file": os.path.basename(PDF_PATH),
        "pdf_sha256": file_sha256(PDF_PATH),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "splitter": "RecursiveCharacterTextSplitter",
        "embedding_model": EMBED_MODEL,
        "collection": COLLECTION,
        "distance": "cosine",
    }


# (a) ------------------------------------------------------------------------
def load_syllabus_text(pdf_path=PDF_PATH):
    """Return the cleaned full text and a list of (start_offset, page_number)."""
    from pypdf import PdfReader

    parts, page_starts, offset = [], [], 0
    for page_no, page in enumerate(PdfReader(pdf_path).pages, 1):
        text = _HEADER_RE.sub("", page.extract_text() or "")  # repeated page header
        text = _PAGENUM_RE.sub("", text)                    # "1/5" page counters
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip() + "\n\n"
        page_starts.append((offset, page_no))
        parts.append(text)
        offset += len(text)
    return "".join(parts), page_starts


def _page_at(offset, page_starts):
    page = page_starts[0][1]
    for start, p in page_starts:
        if offset >= start:
            page = p
    return page


# (b) ------------------------------------------------------------------------
def split_into_chunks(full_text, page_starts):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    # Split the whole document (not page by page) so a section that continues onto the
    # next page stays together; page numbers are recovered from character offsets.
    chunks = splitter.split_documents([Document(page_content=full_text,
                                                metadata={"source": os.path.basename(PDF_PATH)})])
    for i, c in enumerate(chunks):
        start = c.metadata["start_index"]
        p0 = _page_at(start, page_starts)
        p1 = _page_at(start + len(c.page_content) - 1, page_starts)
        c.metadata.update({
            "chunk_id": i,
            "page_start": p0,
            "page_end": p1,
            "pages": str(p0) if p0 == p1 else f"{p0}-{p1}",
        })
    return chunks


# (c) + (e) ------------------------------------------------------------------
def build_index(verbose=True):
    import chromadb
    from langchain_chroma import Chroma

    t0 = time.perf_counter()
    full_text, page_starts = load_syllabus_text()
    chunks = split_into_chunks(full_text, page_starts)
    if verbose:
        print(f"Loaded '{os.path.basename(PDF_PATH)}': {len(page_starts)} pages, "
              f"{len(full_text):,} characters")
        print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, "
              f"chunk_overlap={CHUNK_OVERLAP})")

    # Remove any stale collection, then create a fresh persistent one
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)

    store = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    store.add_documents(chunks, ids=[f"chunk-{c.metadata['chunk_id']:03d}" for c in chunks])

    config = {**current_settings(), "top_k": TOP_K, "chat_model": CHAT_MODEL,
              "n_chunks": len(chunks), "built_at": datetime.now(timezone.utc).isoformat()}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"Embedded with '{EMBED_MODEL}' and saved Chroma index to {PERSIST_DIR}/ "
              f"({time.perf_counter() - t0:.1f}s)")
    return store, config


def load_index(rebuild=False, require_existing=False, verbose=True):
    """Reuse the saved index when it matches the current PDF and settings."""
    import chromadb
    from langchain_chroma import Chroma

    reason = None
    if rebuild:
        reason = "--rebuild requested"
    elif not os.path.exists(CONFIG_PATH):
        reason = "no saved index"
    else:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            saved = json.load(f)
        changed = [k for k, v in current_settings().items() if saved.get(k) != v]
        if changed:
            reason = f"settings or PDF changed ({', '.join(changed)})"

    if reason is None:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        store = Chroma(client=client, collection_name=COLLECTION,
                       embedding_function=get_embeddings())
        count = store._collection.count()
        if count == saved["n_chunks"]:
            if verbose:
                print(f"Reusing saved Chroma index: {PERSIST_DIR}/ ({count} chunks, "
                      f"built {saved['built_at'][:19]}Z, embeddings '{saved['embedding_model']}')")
            return store, saved
        reason = f"index has {count} chunks, expected {saved['n_chunks']}"

    if require_existing:
        raise SystemExit(f"Saved index unusable ({reason}). Run `python rag.py --build` first.")
    if verbose:
        print(f"Building index: {reason}")
    return build_index(verbose)


# (d) + (f) ------------------------------------------------------------------
class SyllabusRAG:
    """Reusable RAG application (also used by rag_eval.py and the Part 4 programs)."""

    def __init__(self, rebuild=False, require_existing=False, k=TOP_K, verbose=True):
        self.store, self.config = load_index(rebuild, require_existing, verbose)
        self.k = k
        self.llm = get_chat_model(temperature=0.0)

    def retrieve(self, question):
        # cosine distance: 0 = identical direction, larger = less similar
        return self.store.similarity_search_with_score(question, k=self.k)

    @staticmethod
    def format_context(hits):
        return "\n\n".join(
            f"[Excerpt {i} | page {doc.metadata['pages']}]\n{doc.page_content.strip()}"
            for i, (doc, _) in enumerate(hits, 1)
        )

    def ask(self, question):
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        messages = [("system", SYSTEM_PROMPT),
                    ("user", USER_TEMPLATE.format(context=self.format_context(hits),
                                                  question=question.strip()))]
        answer = self.llm.invoke(messages).content.strip()
        return {
            "question": question,
            "answer": answer,
            "insufficient_info": answer.startswith(NOT_ENOUGH_INFO),
            "sources": [{"chunk_id": d.metadata["chunk_id"], "pages": d.metadata["pages"],
                         "distance": round(float(s), 4), "text": d.page_content}
                        for d, s in hits],
            "latency_s": round(time.perf_counter() - t0, 2),
        }


def print_result(res, show_chunks=False):
    print(f"\nQ: {res['question']}")
    print(f"A: {res['answer']}")
    srcs = ", ".join(f"p.{s['pages']} (chunk {s['chunk_id']}, dist {s['distance']:.3f})"
                     for s in res["sources"])
    print(f"   Retrieved: {srcs}   [{res['latency_s']:.1f}s]")
    if show_chunks:
        for s in res["sources"]:
            print(f"\n   --- chunk {s['chunk_id']} (p.{s['pages']}) ---\n   "
                  + s["text"].strip().replace("\n", "\n   "))


def main():
    ap = argparse.ArgumentParser(description="RAG over the CPSC 254 syllabus")
    ap.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    ap.add_argument("--build", action="store_true", help="build/verify the index and exit")
    ap.add_argument("--rebuild", action="store_true", help="force re-indexing")
    ap.add_argument("--k", type=int, default=TOP_K, help=f"chunks to retrieve (default {TOP_K})")
    ap.add_argument("--show-chunks", action="store_true", help="print retrieved chunk text")
    args = ap.parse_args()

    if args.build:
        load_index(rebuild=args.rebuild)
        return

    rag = SyllabusRAG(rebuild=args.rebuild, k=args.k)
    if args.question:
        print_result(rag.ask(" ".join(args.question)), args.show_chunks)
        return

    print("\nAsk a question about the CPSC 254 syllabus (empty line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print_result(rag.ask(q), args.show_chunks)


if __name__ == "__main__":
    sys.exit(main())
