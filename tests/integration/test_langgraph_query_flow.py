from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_query_tool_sequence_is_recorded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".refinery").mkdir(parents=True, exist_ok=True)

    response = client.post(
        "/query",
        json={"doc_ids": ["doc-graph"], "query": "Find revenue highlights"},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["tool_sequence"] == ["pageindex_navigate", "semantic_search", "structured_query"]
    trace_id = payload["langsmith_trace_id"]
    assert trace_id and (trace_id.startswith("ls-") or len(trace_id) == 36)
