"""
The R in RAG: a small, dependency-free BM25 retriever over the knowledge base.

The knowledge base (`data/knowledge/notes.jsonl`) holds three kinds of document:

  * `review`   - a critic-style note about one specific song
  * `usecase`  - what listeners actually want in a situation (studying, gym, ...)
  * `glossary` - what a numeric feature means in plain language

Retrieval is deterministic and runs entirely offline, which is what lets the
evaluation harness measure recall and consistency without an API key.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.config import BM25_B, BM25_K1, KNOWLEDGE_JSONL, RETRIEVAL_TOP_K
from src.logging_setup import get_logger

log = get_logger("retrieval")

# Small hand-written stop list. Deliberately short: with a 30-document corpus,
# dropping too much vocabulary hurts recall more than the noise it removes.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could do does for from had has have
    how i if in into is it its me my of on or over so some that the their them
    then there these they this to too up want wants was we were what when which
    while who will with would you your
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """
    Cheap suffix stripper.

    Without it "study" and "studying" are different terms and a query about
    studying misses the study documents entirely - the single biggest recall
    bug found while building the evaluation harness. The rules are crude and
    sometimes produce non-words ("focus" and "focused" both become "focu"),
    which is fine: they only have to be applied consistently to both the query
    and the documents.
    """
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    if len(word) > 4 and word.endswith("ing"):
        word = word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("ly"):
        word = word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase, split, drop stopwords and 1-char tokens, then stem."""
    return [
        stem(tok)
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in STOPWORDS
    ]


@dataclass(frozen=True)
class Document:
    """One retrievable chunk of the knowledge base."""

    doc_id: str
    song_id: Optional[int]
    kind: str
    title: str
    text: str

    @property
    def searchable(self) -> str:
        return f"{self.title} {self.text}"


@dataclass(frozen=True)
class Retrieved:
    """A document plus the score that got it retrieved."""

    document: Document
    score: float

    @property
    def doc_id(self) -> str:
        return self.document.doc_id


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base is missing or malformed."""


def load_documents(path: Path = KNOWLEDGE_JSONL) -> List[Document]:
    """
    Read the JSONL knowledge base.

    A malformed line is logged and skipped rather than killing the run, but a
    missing or entirely unusable file is a hard error - the whole point of the
    system is that answers are grounded in these documents.
    """
    if not path.exists():
        raise KnowledgeBaseError(f"knowledge base not found: {path}")

    documents: List[Document] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                doc = Document(
                    doc_id=str(record["doc_id"]),
                    song_id=(
                        int(record["song_id"])
                        if record.get("song_id") is not None
                        else None
                    ),
                    kind=str(record.get("kind", "note")),
                    title=str(record.get("title", "")),
                    text=str(record["text"]),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("skipping malformed knowledge line %d: %s", line_no, exc)
                continue

            if doc.doc_id in seen:
                log.warning("duplicate doc_id %r on line %d, skipping", doc.doc_id, line_no)
                continue

            seen.add(doc.doc_id)
            documents.append(doc)

    if not documents:
        raise KnowledgeBaseError(f"knowledge base {path} contained no usable documents")

    log.debug("loaded %d knowledge documents from %s", len(documents), path)
    return documents


class BM25Retriever:
    """
    Okapi BM25 over the knowledge base.

    BM25 (rather than embeddings) keeps the project reproducible: no model
    download, no API call, identical scores on every machine and every run.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if not documents:
            raise KnowledgeBaseError("cannot build a retriever over zero documents")

        self.documents: List[Document] = list(documents)
        self.k1 = k1
        self.b = b

        self._tokens: List[List[str]] = [tokenize(d.searchable) for d in self.documents]
        self._counts: List[Counter] = [Counter(toks) for toks in self._tokens]
        self._lengths: List[int] = [len(toks) for toks in self._tokens]
        self._avg_len: float = sum(self._lengths) / len(self._lengths) or 1.0

        n_docs = len(self.documents)
        doc_freq: Counter = Counter()
        for counts in self._counts:
            doc_freq.update(counts.keys())

        # Standard BM25 IDF with the +1 smoothing that keeps it non-negative.
        self._idf: Dict[str, float] = {
            term: math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

        self._index_of: Dict[str, int] = {
            doc.doc_id: i for i, doc in enumerate(self.documents)
        }
        self._by_song: Dict[int, List[Document]] = {}
        for doc in self.documents:
            if doc.song_id is not None:
                self._by_song.setdefault(doc.song_id, []).append(doc)

        log.debug(
            "built BM25 index: %d docs, %d unique terms, avg length %.1f",
            n_docs,
            len(self._idf),
            self._avg_len,
        )

    # -- scoring ------------------------------------------------------------

    def _score_doc(self, query_tokens: Iterable[str], index: int) -> float:
        counts = self._counts[index]
        length = self._lengths[index]
        norm = self.k1 * (1.0 - self.b + self.b * length / self._avg_len)

        score = 0.0
        for term in query_tokens:
            freq = counts.get(term, 0)
            if not freq:
                continue
            score += self._idf.get(term, 0.0) * (freq * (self.k1 + 1.0)) / (freq + norm)
        return score

    def search(self, query: str, k: int = RETRIEVAL_TOP_K) -> List[Retrieved]:
        """Top-k documents for a free-text query, best first."""
        tokens = tokenize(query)
        if not tokens:
            log.warning("retrieval called with an empty/stopword-only query")
            return []

        scored: List[Tuple[float, int]] = [
            (self._score_doc(tokens, i), i) for i in range(len(self.documents))
        ]
        # Tie-break on document order so results are stable across runs.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        hits = [
            Retrieved(document=self.documents[i], score=score)
            for score, i in scored[:k]
            if score > 0.0
        ]
        log.debug(
            "query=%r -> %d hits: %s",
            query,
            len(hits),
            [(h.doc_id, round(h.score, 2)) for h in hits],
        )
        return hits

    # -- per-song evidence --------------------------------------------------

    def notes_for_song(self, song_id: int) -> List[Document]:
        """Every review-style document attached to a specific song."""
        return list(self._by_song.get(song_id, []))

    def evidence_score(self, query: str, song_id: int) -> Tuple[float, List[str]]:
        """
        How strongly the knowledge base supports this song for this query.

        Returns the raw BM25 score summed over the song's own notes, plus the
        doc_ids that contributed. This is the number that feeds back into the
        final ranking (see pipeline.rerank_with_evidence).
        """
        tokens = tokenize(query)
        if not tokens:
            return 0.0, []

        total = 0.0
        used: List[str] = []
        for doc in self._by_song.get(song_id, []):
            score = self._score_doc(tokens, self._index_of[doc.doc_id])
            if score > 0.0:
                total += score
                used.append(doc.doc_id)
        return total, used

    def get(self, doc_id: str) -> Optional[Document]:
        index = self._index_of.get(doc_id)
        return self.documents[index] if index is not None else None


def build_default_retriever(path: Path = KNOWLEDGE_JSONL) -> BM25Retriever:
    """Convenience constructor used by the pipeline, CLI and tests."""
    return BM25Retriever(load_documents(path))


def format_context(hits: Sequence[Retrieved]) -> str:
    """Render retrieved documents into the block handed to the model."""
    if not hits:
        return "(no relevant documents were retrieved)"
    return "\n\n".join(
        f"[{hit.doc_id}] {hit.document.title}\n{hit.document.text}" for hit in hits
    )
