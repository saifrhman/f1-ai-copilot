from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_regulatory_natural_query_returns_503_when_rag_is_unconfigured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post(
        "/api/query/natural",
        json={"query": "Which FIA regulation applies to an unsafe release?"},
    )
    assert response.status_code == 503
    assert "FIA RAG is unavailable" in response.json()["detail"]
