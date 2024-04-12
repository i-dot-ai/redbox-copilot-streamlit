from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field, field_validator


class PersistableModel(BaseModel):
    uuid: UUID = Field(default_factory=uuid4)
    created_datetime: datetime = Field(default_factory=datetime.now)
    creator_user_uuid: Optional[UUID] = None

    @computed_field  # type: ignore[misc]
    @property
    def model_type(self) -> str:
        return self.__class__.__name__

    @field_validator("uuid", "creator_user_uuid")
    @classmethod
    def string_uuid_to_uuid(cls, v: UUID | str) -> UUID:
        if isinstance(v, str):
            return UUID(v)
        return v
