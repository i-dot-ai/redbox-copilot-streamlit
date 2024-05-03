from uuid import UUID

from pydantic import Field

from redbox.models.base import PersistableModel


class Tag(PersistableModel):
    name: str = Field()
    files: set[UUID] = Field(default_factory=set)
