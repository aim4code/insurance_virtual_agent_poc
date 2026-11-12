from datetime import UTC, date, datetime

from fastapi import FastAPI, HTTPException

from crm.models import Callback, CallbackRequest, Handoff, HandoffRequest, PolicySummary
from crm.repository import load_policies

app = FastAPI(title="Aseguradora Demo - mock CRM", version="0.1.0")
POLICIES = load_policies()
CALLBACKS: list[Callback] = []
HANDOFFS: list[Handoff] = []


@app.get("/policies/{policy_number}")
def get_policy(policy_number: str, date_of_birth: date) -> PolicySummary:
    policy = POLICIES.get(policy_number)
    if policy is None or policy.date_of_birth != date_of_birth.isoformat():
        raise HTTPException(status_code=404, detail="identity_check_failed")
    return PolicySummary.model_validate(policy.model_dump())


@app.post("/callbacks", status_code=201)
def create_callback(request: CallbackRequest) -> Callback:
    callback = Callback(
        id=f"CB-{len(CALLBACKS) + 1:04d}",
        created_at=datetime.now(UTC),
        **request.model_dump(),
    )
    CALLBACKS.append(callback)
    return callback


@app.post("/handoffs", status_code=201)
def create_handoff(request: HandoffRequest) -> Handoff:
    handoff = Handoff(
        id=f"HO-{len(HANDOFFS) + 1:04d}",
        created_at=datetime.now(UTC),
        **request.model_dump(),
    )
    HANDOFFS.append(handoff)
    return handoff
