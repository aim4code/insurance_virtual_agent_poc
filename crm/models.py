from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

PolicyStatus = Literal[
    "activa", "suspendida_impago", "cancelada", "pendiente_renovacion"
]


class PolicySummary(BaseModel):
    """What the agent may see after the identity check: no contact data."""

    policy_number: str
    product: Literal["hogar", "auto"]
    plan: str
    status: PolicyStatus
    renewal_date: str | None
    deductible_eur: int
    coverages: list[str]


class Policy(PolicySummary):
    """Full CRM record, including personal data. Never returned by the API."""

    holder_name: str
    date_of_birth: str
    phone: str
    email: str


SPANISH_PHONE = r"^(\+34 ?)?[6789]\d{2} ?\d{3} ?\d{3}$"


class CallbackRequest(BaseModel):
    """What the agent sends to request a callback."""

    phone: str = Field(pattern=SPANISH_PHONE)
    time_window: Literal["mañana", "tarde"]
    reason: str = Field(min_length=3, max_length=500)


class Callback(CallbackRequest):
    """A stored callback, as returned by the API."""

    id: str
    created_at: datetime


class HandoffReason(StrEnum):
    """Reasons for handing off to a human."""

    OUT_OF_SCOPE = "fuera_de_alcance"
    ANGRY_CUSTOMER = "cliente_molesto"
    CLAIM_FILING = "parte_siniestro"
    HUMAN_REQUESTED = "solicitud_cliente"


class HandoffRequest(BaseModel):
    """What the agent sends when it escalates to a human."""

    reason: HandoffReason
    summary: str = Field(min_length=10, max_length=1000)


class Handoff(HandoffRequest):
    """A stored handoff, as returned by the API."""

    id: str
    created_at: datetime
