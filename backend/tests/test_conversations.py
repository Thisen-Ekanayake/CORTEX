import pytest


def _register_and_login(client, email, name="User"):
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "password123", "display_name": name},
    )
    resp = client.post(
        "/api/auth/login",
        json={"email": email, "password": "password123"},
    )
    return resp.json()["access_token"]


@pytest.fixture
def auth_client(client):
    token = _register_and_login(client, "conv@example.com", "Conv Owner")
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def test_create_empty_conversation(auth_client):
    resp = auth_client.post("/api/conversations", json={})
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "New chat"
    assert body["project_id"] is None
    assert body["messages"] == []


def test_append_message_autotitles_and_persists(auth_client):
    cid = auth_client.post("/api/conversations", json={}).json()["id"]

    user_msg = auth_client.post(
        f"/api/conversations/{cid}/messages",
        json={"role": "user", "content": "What is the 2024 travel policy?"},
    )
    assert user_msg.status_code == 201

    auth_client.post(
        f"/api/conversations/{cid}/messages",
        json={
            "role": "assistant",
            "content": "It raised the per-diem cap.",
            "route": "rag_doc",
            "sources": ["policy.pdf"],
        },
    )

    detail = auth_client.get(f"/api/conversations/{cid}").json()
    # Title derived from the first user message.
    assert detail["title"] == "What is the 2024 travel policy?"
    assert len(detail["messages"]) == 2
    assert detail["messages"][0]["role"] == "user"
    assert detail["messages"][1]["route"] == "rag_doc"
    assert detail["messages"][1]["sources"] == ["policy.pdf"]


def test_list_filters_by_project_and_unfiled(auth_client):
    pid = auth_client.post("/api/projects", json={"name": "Folder"}).json()["id"]
    auth_client.post("/api/conversations", json={"title": "in", "project_id": pid})
    auth_client.post("/api/conversations", json={"title": "out"})

    in_project = auth_client.get(f"/api/conversations?project_id={pid}").json()
    assert len(in_project) == 1
    assert in_project[0]["title"] == "in"

    unfiled = auth_client.get("/api/conversations?unfiled=true").json()
    assert len(unfiled) == 1
    assert unfiled[0]["title"] == "out"


def test_move_conversation_into_and_out_of_project(auth_client):
    pid = auth_client.post("/api/projects", json={"name": "Folder"}).json()["id"]
    cid = auth_client.post("/api/conversations", json={"title": "c"}).json()["id"]

    moved = auth_client.patch(f"/api/conversations/{cid}", json={"project_id": pid})
    assert moved.json()["project_id"] == pid

    unfiled = auth_client.patch(f"/api/conversations/{cid}", json={"project_id": None})
    assert unfiled.json()["project_id"] is None


def test_rename_without_touching_project(auth_client):
    pid = auth_client.post("/api/projects", json={"name": "Folder"}).json()["id"]
    cid = auth_client.post(
        "/api/conversations", json={"title": "c", "project_id": pid}
    ).json()["id"]

    # Omitting project_id must leave it unchanged.
    renamed = auth_client.patch(f"/api/conversations/{cid}", json={"title": "renamed"})
    assert renamed.json()["title"] == "renamed"
    assert renamed.json()["project_id"] == pid


def test_delete_conversation(auth_client):
    cid = auth_client.post("/api/conversations", json={}).json()["id"]
    assert auth_client.delete(f"/api/conversations/{cid}").status_code == 204
    assert auth_client.get(f"/api/conversations/{cid}").status_code == 404


def test_conversation_isolation_between_users(auth_client, client):
    cid = auth_client.post("/api/conversations", json={"title": "secret"}).json()["id"]

    other = _register_and_login(client, "intruder2@example.com", "Intruder")
    client.headers.update({"Authorization": f"Bearer {other}"})

    assert client.get("/api/conversations").json() == []
    assert client.get(f"/api/conversations/{cid}").status_code == 404
    assert (
        client.post(
            f"/api/conversations/{cid}/messages",
            json={"role": "user", "content": "hi"},
        ).status_code
        == 404
    )


def test_cannot_file_into_another_users_project(auth_client, client):
    pid = auth_client.post("/api/projects", json={"name": "Mine"}).json()["id"]

    other = _register_and_login(client, "intruder3@example.com", "Intruder")
    client.headers.update({"Authorization": f"Bearer {other}"})

    resp = client.post("/api/conversations", json={"project_id": pid})
    assert resp.status_code == 404
