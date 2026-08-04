"""Tests for the BM25 knowledge-base retriever."""

import pytest

from src.retrieval import (
    BM25Retriever,
    Document,
    KnowledgeBaseError,
    build_default_retriever,
    format_context,
    stem,
    tokenize,
)


@pytest.fixture(scope="module")
def retriever():
    return build_default_retriever()


def make_docs():
    return [
        Document("d1", 1, "review", "Quiet Piano", "A very quiet solo piano lament."),
        Document("d2", 2, "review", "Loud Guitars", "Fast distorted guitars for the gym."),
        Document("d3", None, "usecase", "Studying", "Instrumental music for studying."),
    ]


# -- tokenisation -----------------------------------------------------------


def test_stemmer_collapses_study_and_studying():
    # The bug the evaluation harness caught: without stemming a query about
    # "study" never matched documents that say "studying".
    assert stem("studying") == stem("study")


def test_tokenize_drops_stopwords_and_short_tokens():
    tokens = tokenize("A song for the gym")
    assert "the" not in tokens
    assert "gym" in tokens


# -- search -----------------------------------------------------------------


def test_search_ranks_the_relevant_document_first():
    r = BM25Retriever(make_docs())
    hits = r.search("quiet piano", k=3)
    assert hits[0].doc_id == "d1"


def test_search_returns_no_more_than_k():
    r = BM25Retriever(make_docs())
    assert len(r.search("music", k=1)) <= 1


def test_search_is_deterministic():
    r = BM25Retriever(make_docs())
    first = [h.doc_id for h in r.search("instrumental studying music")]
    second = [h.doc_id for h in r.search("instrumental studying music")]
    assert first == second


def test_search_on_a_stopword_only_query_returns_nothing():
    r = BM25Retriever(make_docs())
    assert r.search("the and of") == []


def test_search_never_returns_zero_scoring_documents():
    r = BM25Retriever(make_docs())
    hits = r.search("piano", k=10)
    assert all(h.score > 0 for h in hits)
    assert len(hits) < 3  # the guitar and studying docs must not appear


# -- per-song evidence ------------------------------------------------------


def test_evidence_score_credits_the_right_song():
    r = BM25Retriever(make_docs())
    score, docs = r.evidence_score("quiet piano lament", song_id=1)
    assert score > 0 and docs == ["d1"]

    score, docs = r.evidence_score("quiet piano lament", song_id=2)
    assert score == 0 and docs == []


def test_notes_for_song_returns_only_that_songs_notes():
    r = BM25Retriever(make_docs())
    assert [d.doc_id for d in r.notes_for_song(2)] == ["d2"]
    assert r.notes_for_song(99) == []


# -- loading and guardrails against a broken corpus -------------------------


def test_empty_corpus_is_rejected():
    with pytest.raises(KnowledgeBaseError):
        BM25Retriever([])


def test_missing_knowledge_file_raises(tmp_path):
    from src.retrieval import load_documents

    with pytest.raises(KnowledgeBaseError):
        load_documents(tmp_path / "nope.jsonl")


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    from src.retrieval import load_documents

    path = tmp_path / "notes.jsonl"
    path.write_text(
        '{"doc_id": "ok", "song_id": 1, "kind": "review", "title": "T", "text": "fine"}\n'
        "{not json at all\n"
        '{"doc_id": "no-text", "song_id": 2}\n'
        "\n",
        encoding="utf-8",
    )
    docs = load_documents(path)
    assert [d.doc_id for d in docs] == ["ok"]


# -- the real knowledge base ------------------------------------------------


def test_real_knowledge_base_loads_and_has_all_three_kinds(retriever):
    kinds = {d.kind for d in retriever.documents}
    assert {"review", "usecase", "glossary"} <= kinds


def test_every_catalog_song_has_at_least_one_note(retriever):
    from src.recommender import load_songs

    for song in load_songs():
        assert retriever.notes_for_song(song["id"]), f"no note for {song['title']}"


def test_format_context_labels_documents_with_their_ids(retriever):
    hits = retriever.search("music for the gym", k=2)
    rendered = format_context(hits)
    for hit in hits:
        assert f"[{hit.doc_id}]" in rendered


def test_format_context_handles_no_hits():
    assert "no relevant documents" in format_context([])
