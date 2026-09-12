from types import SimpleNamespace

from core_modules.rule_checker.fia_rag_agent import (
    DECLINE_ANSWER,
    FIAKnowledgeBase,
    RetrievedPassage,
)


def _passage(score=0.9):
    return RetrievedPassage(
        text="Article 12.4 states that the car must remain within the defined limit.",
        score=score,
        source="sporting_regulations.pdf",
        page=42,
        chunk_id="chunk-1",
    )


def test_retrieval_is_independent_from_generation():
    kb = FIAKnowledgeBase(openai_api_key="test-key")
    kb.initialize = lambda: None

    class FakeEmbeddings:
        def embed_query(self, question):
            assert question == "What does Article 12.4 require?"
            return [0.1, 0.2]

    class FakeQdrant:
        def search(self, **kwargs):
            assert kwargs["limit"] == 3
            return [
                SimpleNamespace(
                    score=0.91,
                    id="chunk-1",
                    payload={
                        "text": "Article 12.4 requires X.",
                        "source": "sporting_regulations.pdf",
                        "page": 4,
                        "chunk_id": "chunk-1",
                    },
                )
            ]

    kb.embeddings = FakeEmbeddings()
    kb.qdrant = FakeQdrant()

    passages = kb.retrieve("What does Article 12.4 require?", top_k=3)

    assert len(passages) == 1
    assert passages[0].score == 0.91
    assert passages[0].page == 5
    assert passages[0].source == "sporting_regulations.pdf"


def test_generation_declines_when_retrieval_is_below_threshold():
    kb = FIAKnowledgeBase(openai_api_key="test-key", min_score=0.5)
    kb.llm = SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(AssertionError("LLM must not run")))

    answer = kb.generate_answer("question", [_passage(score=0.2)])

    assert answer == DECLINE_ANSWER


def test_generation_prompt_contains_only_retrieved_evidence_and_source_label():
    kb = FIAKnowledgeBase(openai_api_key="test-key", min_score=0.3)

    class FakeLLM:
        def __init__(self):
            self.prompt = None

        def invoke(self, prompt):
            self.prompt = prompt
            return SimpleNamespace(content="Article 12.4 requires this. [S1]")

    fake_llm = FakeLLM()
    kb.llm = fake_llm

    answer = kb.generate_answer("What does the rule require?", [_passage()])

    assert answer == "Article 12.4 requires this. [S1]"
    assert "sporting_regulations.pdf, page 42" in fake_llm.prompt
    assert "Article 12.4 states" in fake_llm.prompt
    assert "Use ONLY the retrieved regulation excerpts" in fake_llm.prompt


def test_query_returns_evidence_and_citations():
    kb = FIAKnowledgeBase(openai_api_key="test-key", min_score=0.3)
    kb.retrieve = lambda _: [_passage()]
    kb.generate_answer = lambda question, passages: "Article 12.4 applies here. [S1]"

    result = kb.query("What applies?")

    assert result["grounded"] is True
    assert result["top_retrieval_score"] == 0.9
    assert result["citations"] == ["[S1]"]
    assert result["referenced_rules"] == ["Article 12.4"]
    assert result["retrieved_passages"][0]["chunk_id"] == "chunk-1"


def test_query_declines_without_evidence():
    kb = FIAKnowledgeBase(openai_api_key="test-key", min_score=0.3)
    kb.retrieve = lambda _: []

    result = kb.query("What is the FIA teleportation rule?")

    assert result["answer"] == DECLINE_ANSWER
    assert result["grounded"] is False
    assert result["retrieved_passages"] == []


def test_rule_extraction_handles_nested_rule_numbers():
    rules = FIAKnowledgeBase._extract_rules(
        "Article 12.4.1 and Section 3.2 apply; Article 12.4.1 is repeated."
    )

    assert rules == ["Article 12.4.1", "Section 3.2"]
