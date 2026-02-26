import pytest

@pytest.fixture
def auth_header(client):
    client.post(
        "/api/auth/register",
        json={"email": "user@example.com", "password": "password123", "display_name": "User One"}
    )
    response = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "password123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_get_profile(client, auth_header):
    response = client.get("/api/users/me", headers=auth_header)
    assert response.status_code == 200
    assert response.json()["email"] == "user@example.com"

def test_update_profile(client, auth_header):
    response = client.put(
        "/api/users/me",
        headers=auth_header,
        json={"display_name": "Updated Name"}
    )
    assert response.status_code == 200
    assert response.json()["display_name"] == "Updated Name"

def test_update_password(client, auth_header):
    response = client.put(
        "/api/users/me/password",
        headers=auth_header,
        json={"current_password": "password123", "new_password": "newpassword123"}
    )
    assert response.status_code == 204
    
    # Verify login with new password
    response = client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "newpassword123"}
    )
    assert response.status_code == 200

def test_update_preferences(client, auth_header):
    response = client.put(
        "/api/users/me/preferences",
        headers=auth_header,
        json={"dark_mode": True, "compact_sidebar": True}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["dark_mode"] is True
    assert data["compact_sidebar"] is True
