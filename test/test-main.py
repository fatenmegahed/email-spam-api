from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Email Spam Classification API is running"
    }


def test_predict_spam():
    response = client.post(
        "/predict",
        json={
            "email": "Congratulations! You won a free prize!"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "email" in data
    assert "prediction" in data
    assert data["prediction"] in ["spam", "ham"]


def test_predict_ham():
    response = client.post(
        "/predict",
        json={
            "email": "Hi, please send me the report for today's meeting."
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "email" in data
    assert "prediction" in data
    assert data["prediction"] in ["spam", "ham"]


def test_predict_missing_email():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422