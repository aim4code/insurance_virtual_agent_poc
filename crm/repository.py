import json

from pathlib import Path
from crm.models import Policy

DATA_FILE = Path(__file__).parent / "data" / "policies.json"


def load_policies(path: Path = DATA_FILE) -> dict[str, Policy]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {p["policy_number"]: Policy(**p) for p in raw["policies"]}
