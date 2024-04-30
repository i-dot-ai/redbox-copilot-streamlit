from typing import Optional
from uuid import UUID

from redbox.models.base import PersistableModel


class User(PersistableModel):
    uuid: UUID
    name: Optional[str]
    email: Optional[str]
    department: str
    role: str
    preferred_language: str

    def str_llm(self) -> str:
        return "\n".join([f"{k}: {v}" for k, v in self.model_dump().items() if k not in ("uuid", "email", "name")])
