from typing import Literal
from pydantic import BaseModel

PolicyStatus = Literal["activa", "suspendida_impago",
                       "cancelada", "pendiente_renovacion"]


class Policy(BaseModel):
    policy_number: str
    product: Literal["hogar", "auto"]
    plan: str
    status: PolicyStatus
    holder_name: str
    date_of_birth: str
    phone: str
    email: str
    renewal_date: str | None
    deductible_eur: int
    coverages: list[str]
