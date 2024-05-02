from typing import Optional
from uuid import UUID

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from pydantic import field_serializer, field_validator

from redbox.models.base import PersistableModel


class SummaryTask(PersistableModel):
    id: str
    title: str
    # langchain.prompts.PromptTemplate needs pydantic v1, breaks
    # https://python.langchain.com/docs/guides/pydantic_compatibility
    prompt_template: object

    @field_serializer("prompt_template")
    def serialise_prompt(self, prompt_template: PromptTemplate, _info):
        if isinstance(prompt_template, PromptTemplate):
            return prompt_template.dict()
        else:
            return prompt_template


class SummaryTaskComplete(SummaryTask):
    id: str
    title: str
    file_uuids: list[UUID]
    response_text: str
    # langchain_core.documents.Document needs pydantic v1, breaks
    # https://python.langchain.com/docs/guides/pydantic_compatibility
    sources: Optional[list[object]]

    @field_validator("sources")
    @classmethod
    def is_document(cls, v: list[Document]) -> list[Document]:
        """Custom validator for using Pydantic v1 object in model."""
        assert all(isinstance(doc, Document) for doc in v)
        return v

    @field_serializer("sources")
    def serialise_doc(self, sources: list[Document], _info):
        return [source.dict() for source in sources]


class SummaryBase(PersistableModel):
    file_uuids: list[UUID]
    file_hash: int


class Summary(SummaryBase):
    tasks: list[SummaryTask]


class SummaryComplete(SummaryBase):
    tasks: list[SummaryTaskComplete]
