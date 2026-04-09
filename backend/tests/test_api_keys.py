import pytest

@pytest.fixture
def auth_client(client):
    client.post(
        "/api/auth/register",
        json={"email": "keys@example.com", "password": "password123", "display_name": "Key Master"}
    )
    response = client.post(
        "/api/auth/login",
        json={"email": "keys@example.com", "password": "password123"}
    )
    token = response.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client

def test_create_api_key(auth_client):
    response = auth_client.post("/api/api-keys", json={"label": "Secret Key"})
    assert response.status_code == 201
    data = response.json()
    assert "key" in data
    assert data["label"] == "Secret Key"
    assert data["key"].startswith("sk_live_")

def test_list_api_keys(auth_client):
    auth_client.post("/api/api-keys", json={"label": "Key 1"})
    auth_client.post("/api/api-keys", json={"label": "Key 2"})
    
    response = auth_client.get("/api/api-keys")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 2

def test_revoke_api_key(auth_client):
    key = auth_client.post("/api/api-keys", json={"label": "To Revoke"}).json()
    key_id = key["id"]
    
    response = auth_client.delete(f"/api/api-keys/{key_id}")
    assert response.status_code == 204
    
    # Verify revocation
    keys = auth_client.get("/api/api-keys").json()
    revoked = next(k for k in keys if k["id"] == key_id)
    assert revoked["is_active"] is False
    assert revoked["revoked_at"] is not None
