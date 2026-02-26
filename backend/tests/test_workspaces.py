import pytest

@pytest.fixture
def auth_client(client):
    client.post(
        "/api/auth/register",
        json={"email": "ws@example.com", "password": "password123", "display_name": "WS Admin"}
    )
    response = client.post(
        "/api/auth/login",
        json={"email": "ws@example.com", "password": "password123"}
    )
    token = response.json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client

def test_list_workspaces(auth_client):
    # Registration should have auto-created one workspace
    response = auth_client.get("/api/workspaces")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "WS Admin's Workspace" in data[0]["name"]

def test_get_members(auth_client):
    workspaces = auth_client.get("/api/workspaces").json()
    ws_id = workspaces[0]["id"]
    
    response = auth_client.get(f"/api/workspaces/{ws_id}/members")
    assert response.status_code == 200
    members = response.json()
    assert len(members) == 1
    assert members[0]["display_name"] == "WS Admin"
    assert members[0]["role"] == "owner"

def test_invite_member(auth_client, client):
    # Register another user to invite
    client.post(
        "/api/auth/register",
        json={"email": "invitee@example.com", "password": "password123", "display_name": "Invitee"}
    )
    
    workspaces = auth_client.get("/api/workspaces").json()
    ws_id = workspaces[0]["id"]
    
    response = auth_client.post(
        f"/api/workspaces/{ws_id}/members",
        json={"email": "invitee@example.com", "role": "member"}
    )
    assert response.status_code == 201
    assert response.json()["email"] == "invitee@example.com"
