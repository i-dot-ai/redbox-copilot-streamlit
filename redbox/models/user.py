from typing import Optional
from uuid import UUID

from redbox.models.base import PersistableModel

from pydantic import EmailStr


class User(PersistableModel):
    name: Optional[str]
    email: Optional[EmailStr]
    uuid: UUID
    department: Optional[str]
    role: Optional[str]
    preferred_language: Optional[str]

    def summary(self) -> str:
        return (
            f"Name: {self.name} \n"
            f"Department: {self.department} \n"
            f"Role: {self.role} \n"
            f"Preferred language: {self.preferred_language} \n"
        )
