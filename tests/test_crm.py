import pytest
from fastapi.testclient import TestClient

from crm.app import CALLBACKS, HANDOFFS, app

VALID_POLICY = "POL-H-100001"
VALID_DOB = "1985-03-14"
VALID_CALLBACK = {
    "phone": "600 000 101",
    "time_window": "tarde",
    "reason": "Dudas sobre la renovación",
}
VALID_HANDOFF = {
    "reason": "parte_siniestro",
    "summary": "El cliente quiere dar parte de un accidente con su coche.",
}


@pytest.fixture
def client() -> TestClient:
    CALLBACKS.clear()
    HANDOFFS.clear()
    return TestClient(app)


# GET /policies


def test_get_policy_with_correct_dob_returns_summary(client: TestClient) -> None:
    response = client.get(
        f"/policies/{VALID_POLICY}", params={"date_of_birth": VALID_DOB}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["policy_number"] == VALID_POLICY
    assert body["status"] == "activa"


def test_get_policy_never_returns_personal_data(client: TestClient) -> None:
    response = client.get(
        f"/policies/{VALID_POLICY}", params={"date_of_birth": VALID_DOB}
    )
    body = response.json()
    for field in ["holder_name", "date_of_birth", "phone", "email"]:
        assert field not in body


@pytest.mark.parametrize(
    ("policy_number", "date_of_birth"),
    [
        (VALID_POLICY, "1985-03-15"),  # wrong date of birth
        ("POL-H-999999", VALID_DOB),  # unknown policy
    ],
)
def test_get_policy_identity_failures_look_the_same(
    client: TestClient, policy_number: str, date_of_birth: str
) -> None:
    response = client.get(
        f"/policies/{policy_number}", params={"date_of_birth": date_of_birth}
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "identity_check_failed"}


def test_get_policy_rejects_badly_formatted_dob(client: TestClient) -> None:
    response = client.get(
        f"/policies/{VALID_POLICY}", params={"date_of_birth": "14/03/1985"}
    )
    assert response.status_code == 422


# POST /callbacks


def test_create_callback_returns_201_and_sequential_ids(client: TestClient) -> None:
    first = client.post("/callbacks", json=VALID_CALLBACK)
    second = client.post("/callbacks", json=VALID_CALLBACK)

    assert first.status_code == 201
    assert first.json()["id"] == "CB-0001"
    assert second.json()["id"] == "CB-0002"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phone", "12345"),
        ("phone", "500 000 101"),
        ("time_window", "noche"),
        ("reason", ""),
    ],
)
def test_create_callback_rejects_invalid_input(
    client: TestClient, field: str, value: str
) -> None:
    response = client.post("/callbacks", json={**VALID_CALLBACK, field: value})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]


# POST /handoffs


def test_create_handoff_returns_201(client: TestClient) -> None:
    response = client.post("/handoffs", json=VALID_HANDOFF)

    assert response.status_code == 201
    assert response.json()["id"] == "HO-0001"
    assert response.json()["reason"] == "parte_siniestro"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", "otro"),
        ("summary", "hola"),
    ],
)
def test_create_handoff_rejects_invalid_input(
    client: TestClient, field: str, value: str
) -> None:
    response = client.post("/handoffs", json={**VALID_HANDOFF, field: value})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", field]
