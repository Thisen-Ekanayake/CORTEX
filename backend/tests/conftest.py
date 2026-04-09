import pytest
from sqlalchemy import make_url
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import get_db
from app.database import Base, engine, SessionLocal

# Use the cortex_test database for tests
TEST_DATABASE_URL = "postgresql://thisen-ekanayake@/cortex_test?host=/tmp"

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after session
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db():
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db):
    def override_get_db():
        try:
            yield db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
