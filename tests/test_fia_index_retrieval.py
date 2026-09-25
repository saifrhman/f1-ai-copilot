"""Indexing and retrieval against a real embedded Qdrant store (only the embedding API is replaced)."""

import pytest
from qdrant_client import QdrantClient, models

from core_modules.rule_checker.fia_rag import (
    ChunkingConfig,
    EmbeddingConfig,
    FIARegulationRAG,
    IndexNotReadyError,
    QdrantConfig,
    RAGConfigurationError,
    RAGSettings,
    RetrievalConfig,
    VectorStoreError,
)
from core_modules.rule_checker.fia_rag.index import close_qdrant_clients, create_qdrant_client
from tests.helpers import HashingEmbeddings, ScriptedChatModel, fia_page, write_pdf

PIT_LANE = (
    "B1.6 Pit Lane Speed\nB1.6.3 Driving in the Pit Entry Road, Pit Lane and Pit Exit Road "
    "a. A speed limit of 80km/h will be imposed in the pit lane during all sessions."
)
UNSAFE_RELEASE = (
    "B4.2 Unsafe Release\nB4.2.1 A car must not be released from its pit stop position in an unsafe "
    "condition. Competitors are responsible for releasing cars only when it is safe."
)
FUEL_FLOW = "C5.4 Fuel Flow\nC5.4.2 The fuel mass flow must not exceed one hundred kilograms per hour above 10500 rpm."
REAR_WING = "C3.9 Rear Wing\nC3.9.1 The rear wing flap position may be adjusted by the driver only when the adjustable wing is enabled."


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "fia_docs"
    write_pdf(folder / "section_b_sporting.pdf", [fia_page(1, PIT_LANE), None, fia_page(3, UNSAFE_RELEASE)])
    write_pdf(folder / "section_c_technical.pdf", [FUEL_FLOW, REAR_WING])
    return folder


def make_settings(docs, tmp_path, **overrides):
    values = dict(
        docs_path=docs,
        chunking=ChunkingConfig(chunk_size=400, chunk_overlap=40),
        embedding=EmbeddingConfig(model="hashing-test-512", batch_size=2),
        retrieval=RetrievalConfig(top_k=4, min_score=0.2),
        qdrant=QdrantConfig(collection="fia_test", path=tmp_path / "qdrant"),
    )
    values.update(overrides)
    return RAGSettings(**values)


@pytest.fixture
def qdrant():
    client = QdrantClient(":memory:")
    yield client
    client.close()


def make_rag(docs, tmp_path, qdrant, embeddings=None, llm=None, **overrides):
    return FIARegulationRAG(
        make_settings(docs, tmp_path, **overrides),
        embeddings=embeddings or HashingEmbeddings(),
        llm=llm,
        qdrant_client=qdrant,
    )


# ------------------------------------------------------------------ indexing


def test_index_build_is_idempotent_and_does_not_reembed(docs, tmp_path, qdrant):
    embeddings = HashingEmbeddings()
    rag = make_rag(docs, tmp_path, qdrant, embeddings)
    first = rag.build_index()
    calls_after_first = embeddings.document_calls
    second = rag.build_index()
    third = make_rag(docs, tmp_path, qdrant, embeddings).build_index()

    assert first.status == "rebuilt" and first.chunks == 4
    assert second.status == third.status == "up_to_date"
    assert embeddings.document_calls == calls_after_first == 2  # 4 chunks / batch_size 2
    assert qdrant.count("fia_test", exact=True).count == 4


def test_forced_rebuild_replaces_instead_of_duplicating(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant)
    rag.build_index()
    ids_before = {p.id for p in qdrant.scroll("fia_test", limit=100)[0]}
    rag.build_index(force=True)
    ids_after = {p.id for p in qdrant.scroll("fia_test", limit=100)[0]}
    assert qdrant.count("fia_test", exact=True).count == 4
    assert ids_before == ids_after


def test_changed_chunking_makes_index_stale_until_rebuilt(docs, tmp_path, qdrant):
    make_rag(docs, tmp_path, qdrant).build_index()
    rechunked = make_rag(docs, tmp_path, qdrant, chunking=ChunkingConfig(chunk_size=120, chunk_overlap=0))
    assert rechunked.status()["index"]["status"] == "stale"
    with pytest.raises(IndexNotReadyError, match="different documents, chunking"):
        rechunked.retrieve("pit lane speed limit")
    report = rechunked.build_index()
    assert report.status == "rebuilt" and report.chunks > 4
    assert qdrant.count("fia_test", exact=True).count == report.chunks  # old chunks are gone


def test_new_or_changed_documents_make_index_stale(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant)
    rag.build_index()
    write_pdf(docs / "section_f_operational.pdf", ["F2.1 Shutdown periods apply to every competitor team."])
    assert rag.status()["index"]["status"] == "stale"
    assert rag.build_index().chunks == 5


def test_changed_embedding_model_makes_index_stale(docs, tmp_path, qdrant):
    make_rag(docs, tmp_path, qdrant).build_index()
    other = make_rag(docs, tmp_path, qdrant, embedding=EmbeddingConfig(model="another-model", batch_size=2))
    assert other.status()["index"]["status"] == "stale"


def test_incomplete_index_is_detected(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant)
    rag.build_index()
    victim = qdrant.scroll("fia_test", limit=1)[0][0].id
    qdrant.delete("fia_test", points_selector=models.PointIdsList(points=[victim]))
    assert rag.status()["index"]["status"] == "incomplete"
    with pytest.raises(IndexNotReadyError, match="incomplete"):
        rag.retrieve("pit lane")
    assert rag.build_index().status == "rebuilt"


def test_missing_index_is_reported_not_built_implicitly(docs, tmp_path, qdrant):
    embeddings = HashingEmbeddings()
    rag = make_rag(docs, tmp_path, qdrant, embeddings)
    status = rag.status()
    assert status["index"]["status"] == "missing" and status["ready"] is False
    with pytest.raises(IndexNotReadyError, match="build_fia_index.py"):
        rag.answer("What is the pit lane speed limit?")
    assert embeddings.document_calls == 0


def test_query_vectors_from_a_different_dimension_are_rejected(docs, tmp_path, qdrant):
    make_rag(docs, tmp_path, qdrant, HashingEmbeddings(dimension=512)).build_index()
    mismatched = make_rag(docs, tmp_path, qdrant, HashingEmbeddings(dimension=256))
    with pytest.raises(VectorStoreError, match="different embedding model"):
        mismatched.retrieve("pit lane speed")


def test_local_persistent_index_survives_restart(docs, tmp_path):
    settings = make_settings(docs, tmp_path)
    rag = FIARegulationRAG(settings, embeddings=HashingEmbeddings())
    assert rag.build_index().status == "rebuilt"
    close_qdrant_clients()  # simulate a process restart

    embeddings = HashingEmbeddings()
    restarted = FIARegulationRAG(settings, embeddings=embeddings)
    assert restarted.build_index().status == "up_to_date"
    assert embeddings.document_calls == 0
    assert restarted.retrieve("pit lane speed limit").passages[0].source == "section_b_sporting.pdf"
    close_qdrant_clients()


def test_one_embedded_client_is_shared_per_storage_path(tmp_path):
    config = QdrantConfig(collection="x", path=tmp_path / "q")
    assert create_qdrant_client(config) is create_qdrant_client(config)
    close_qdrant_clients()


def test_missing_api_key_fails_instead_of_using_fake_embeddings(docs, tmp_path, qdrant):
    rag = FIARegulationRAG(make_settings(docs, tmp_path), qdrant_client=qdrant)
    with pytest.raises(RAGConfigurationError, match="OPENAI_API_KEY"):
        rag.build_index()


# ------------------------------------------------------------------ retrieval


def test_retrieval_ranks_relevant_passage_first_with_metadata(docs, tmp_path, qdrant):
    llm = ScriptedChatModel()
    rag = make_rag(docs, tmp_path, qdrant, llm=llm)
    rag.build_index()
    result = rag.retrieve("What is the speed limit in the pit lane?")

    best = result.passages[0]
    assert "80km/h" in best.text
    assert (best.source, best.page, best.page_label, best.nearest_rule) == ("section_b_sporting.pdf", 1, "B1", "B1.6")
    assert "B1.6.3" in best.rule_ids
    scores = [p.score for p in result.passages + result.below_threshold]
    assert scores == sorted(scores, reverse=True)
    assert llm.calls == []  # retrieval never calls the answer model


def test_top_k_controls_retrieval_depth(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant, retrieval=RetrievalConfig(top_k=4, min_score=0.0))
    rag.build_index()
    assert len(rag.retrieve("pit lane speed", top_k=1).passages) == 1
    assert len(rag.retrieve("pit lane speed", top_k=3).passages) == 3
    assert len(rag.retrieve("pit lane speed").passages) == 4


@pytest.mark.parametrize("top_k", [0, -1, 51, True])
def test_invalid_top_k_is_a_caller_error_not_an_outage(docs, tmp_path, qdrant, top_k):
    rag = make_rag(docs, tmp_path, qdrant)
    rag.build_index()
    with pytest.raises(ValueError) as caught:
        rag.retrieve("pit lane", top_k=top_k)
    assert not isinstance(caught.value, RAGConfigurationError)


def test_threshold_keeps_only_passages_at_or_above_min_score(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant, retrieval=RetrievalConfig(top_k=4, min_score=0.0))
    rag.build_index()
    everything = rag.retrieve("pit lane speed limit")
    cut = sorted(p.score for p in everything.passages)[-2]  # keep the two best
    filtered = rag.retrieve("pit lane speed limit", min_score=cut)

    assert len(everything.passages) == 4 and not everything.below_threshold
    assert all(p.score >= cut for p in filtered.passages) and len(filtered.passages) == 2
    assert all(p.score < cut for p in filtered.below_threshold) and len(filtered.below_threshold) == 2


def test_unrelated_question_yields_no_evidence(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant)
    rag.build_index()
    result = rag.retrieve("chocolate cake recipe with eggs and butter")
    assert result.passages == []
    assert result.top_score < 0.2


def test_duplicate_passages_are_collapsed(tmp_path, qdrant):
    folder = tmp_path / "dup_docs"
    write_pdf(folder / "a.pdf", [PIT_LANE])
    write_pdf(folder / "b.pdf", ["Front matter of another document that is different.", PIT_LANE])
    rag = make_rag(folder, tmp_path, qdrant, retrieval=RetrievalConfig(top_k=3, min_score=0.0))
    rag.build_index()
    result = rag.retrieve("pit lane speed limit 80km/h")
    texts = [p.text for p in result.passages]
    assert result.duplicates_removed == 1
    assert len(texts) == len(set(texts))


def test_empty_question_is_rejected(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant)
    rag.build_index()
    with pytest.raises(ValueError):
        rag.retrieve("   ")


def test_embedding_cache_avoids_repeat_provider_calls(docs, tmp_path, qdrant):
    cache = tmp_path / "cache" / "embeddings.sqlite3"
    settings = make_settings(docs, tmp_path, embedding=EmbeddingConfig(model="hashing-test-512", batch_size=2, cache_path=cache))
    first = FIARegulationRAG(settings, embeddings=HashingEmbeddings(), qdrant_client=qdrant)
    first.build_index()
    assert first.embedder().provider_requests == 2

    fresh = HashingEmbeddings()
    second = FIARegulationRAG(settings, embeddings=fresh, qdrant_client=qdrant)
    second.build_index(force=True)  # full rebuild served from the cache
    second.retrieve("pit lane speed limit")
    second.retrieve("pit lane speed limit")
    assert fresh.document_calls == 0 and fresh.query_calls == 1
    assert qdrant.count("fia_test", exact=True).count == 4

    other_model = FIARegulationRAG(
        make_settings(docs, tmp_path, embedding=EmbeddingConfig(model="other-model", batch_size=2, cache_path=cache)),
        embeddings=HashingEmbeddings(),
        qdrant_client=qdrant,
    )
    other_model.build_index()
    assert other_model.embedder().provider_requests == 2  # cache entries are per model


def test_threshold_can_be_reapplied_without_new_search(docs, tmp_path, qdrant):
    embeddings = HashingEmbeddings()
    rag = make_rag(docs, tmp_path, qdrant, embeddings, retrieval=RetrievalConfig(top_k=4, min_score=0.0))
    rag.build_index()
    everything = rag.retrieve("pit lane speed limit")
    strict = everything.with_threshold(0.99)
    assert strict.passages == [] and len(strict.below_threshold) == 4
    assert everything.with_threshold(0.0).passages == everything.passages
    assert embeddings.query_calls == 1


# ------------------------------------------------------- atomic rebuild (alias swap)


class ObservedQdrant(QdrantClient):
    """Real in-memory Qdrant that runs ``observer`` before every write, as a concurrent reader would."""

    def __init__(self):
        super().__init__(":memory:")
        self.observer = None
        self.seen = []

    def _observe(self):
        if self.observer is not None:
            try:
                self.seen.append(self.observer())
            except Exception as exc:  # what the reader would have answered its client
                self.seen.append(type(exc).__name__)

    def delete_collection(self, *args, **kwargs):
        self._observe()
        return super().delete_collection(*args, **kwargs)

    def upsert(self, *args, **kwargs):
        self._observe()
        return super().upsert(*args, **kwargs)

    def update_collection_aliases(self, *args, **kwargs):
        self._observe()
        return super().update_collection_aliases(*args, **kwargs)


class FailingUpserts(QdrantClient):
    """Real in-memory Qdrant whose upserts fail once ``fail`` is set (storage outage during a rebuild)."""

    def __init__(self):
        super().__init__(":memory:")
        self.fail = False

    def upsert(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("simulated Qdrant outage")
        return super().upsert(*args, **kwargs)


def collection_names(client):
    return sorted(c.name for c in client.get_collections().collections)


def test_forced_rebuild_never_exposes_an_empty_or_partial_index(docs, tmp_path):
    client = ObservedQdrant()
    rag = make_rag(docs, tmp_path, client)
    rag.build_index()
    reader = make_rag(docs, tmp_path, client)  # a second API process reading the same index
    client.observer = lambda: (
        reader.status()["index"]["status"],
        reader.retrieve("pit lane speed limit").passages[0].source,
    )
    assert rag.build_index(force=True).status == "rebuilt"
    client.observer = None
    assert client.seen and set(client.seen) == {("current", "section_b_sporting.pdf")}
    client.close()


def test_rebuild_swaps_alias_to_a_new_collection_and_drops_the_old_one(docs, tmp_path, qdrant):
    rag = make_rag(docs, tmp_path, qdrant)
    rag.build_index()
    index = rag.index()
    first, first_glossary = index.alias_target("fia_test"), index.alias_target("fia_test_glossary")
    assert first.startswith("fia_test__") and first_glossary.startswith("fia_test_glossary__")
    rag.build_index(force=True)
    second = index.alias_target("fia_test")
    assert second != first
    assert collection_names(qdrant) == sorted([second, index.alias_target("fia_test_glossary")])
    assert rag.retrieve("pit lane speed limit").passages[0].source == "section_b_sporting.pdf"


def test_failed_rebuild_keeps_the_previous_index_serving(docs, tmp_path):
    client = FailingUpserts()
    rag = make_rag(docs, tmp_path, client)
    rag.build_index()
    before = collection_names(client)
    client.fail = True
    with pytest.raises(VectorStoreError, match="simulated Qdrant outage"):
        rag.build_index(force=True)
    client.fail = False
    assert collection_names(client) == before  # the unfinished collection was removed
    assert rag.status()["index"]["status"] == "current"
    assert rag.retrieve("pit lane speed limit").passages[0].source == "section_b_sporting.pdf"
    client.close()


def test_index_built_before_aliases_is_migrated_on_rebuild(docs, tmp_path, qdrant):
    qdrant.create_collection("fia_test", vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))
    qdrant.upsert("fia_test", points=[models.PointStruct(id=1, vector=[1.0] + [0.0] * 511, payload={"text": "old"})])
    rag = make_rag(docs, tmp_path, qdrant)
    assert rag.status()["index"]["status"] == "stale"
    assert rag.build_index().status == "rebuilt"
    assert "fia_test" not in collection_names(qdrant)
    assert rag.index().alias_target("fia_test").startswith("fia_test__")
    assert rag.retrieve("pit lane speed limit").passages[0].source == "section_b_sporting.pdf"
