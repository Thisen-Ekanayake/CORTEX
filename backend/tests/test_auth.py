import pytest

def test_register_user(client):
    response = client.post(
        "/api/auth/register",
        json={"email": "test@example.com", "password": "password123", "display_name": "Test User"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["display_name"] == "Test User"
    assert "id" in data
    assert data["avatar_initials"] == "TU"

def test_login_user(client):
    # Register first
    client.post(
        "/api/auth/register",
        json={"email": "login@example.com", "password": "password123", "display_name": "Login User"}
    )
    
    # Login
    response = client.post(
        "/api/auth/login",
        json={"email": "login@example.com", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_invalid_credentials(client):
    response = client.post(
        "/api/auth/login",
        json={"email": "wrong@example.com", "password": "wrong"}
    )
    assert response.status_code == 401
