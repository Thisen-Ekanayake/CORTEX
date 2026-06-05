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
    token = _register_and_login(client, "proj@example.com", "Proj Owner")
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def test_create_and_list_project(auth_client):
    resp = auth_client.post("/api/projects", json={"name": "Project Alpha"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Project Alpha"
    assert body["icon"] == "◰"

    listing = auth_client.get("/api/projects")
    assert listing.status_code == 200
    assert len(listing.json()) == 1


def test_rename_project(auth_client):
    pid = auth_client.post("/api/projects", json={"name": "Old"}).json()["id"]
    resp = auth_client.patch(f"/api/projects/{pid}", json={"name": "New", "icon": "◳"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"
    assert resp.json()["icon"] == "◳"


def test_delete_project_unfiles_conversations(auth_client):
    pid = auth_client.post("/api/projects", json={"name": "Temp"}).json()["id"]
    cid = auth_client.post(
        "/api/conversations", json={"title": "c", "project_id": pid}
    ).json()["id"]

    assert auth_client.delete(f"/api/projects/{pid}").status_code == 204

    # Conversation survives, now unfiled.
    conv = auth_client.get(f"/api/conversations/{cid}").json()
    assert conv["project_id"] is None


def test_project_isolation_between_users(auth_client, client):
    pid = auth_client.post("/api/projects", json={"name": "Private"}).json()["id"]

    other = _register_and_login(client, "intruder@example.com", "Intruder")
    client.headers.update({"Authorization": f"Bearer {other}"})

    assert client.get("/api/projects").json() == []
    assert client.patch(f"/api/projects/{pid}", json={"name": "x"}).status_code == 404
    assert client.delete(f"/api/projects/{pid}").status_code == 404


def test_projects_require_auth(client):
    assert client.get("/api/projects").status_code == 401
