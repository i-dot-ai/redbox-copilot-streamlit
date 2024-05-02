from typing import Optional
from uuid import UUID

from langchain.prompts import PromptTemplate
from pydantic import field_serializer, field_validator

from redbox.models.base import PersistableModel
from redbox.models.chat import ChatSource


class SummaryTask(PersistableModel):
    id: str
    title: str
    # langchain.prompts.PromptTemplate needs pydantic v1, breaks
    # https://python.langchain.com/docs/guides/pydantic_compatibility
    prompt_template: object

    @field_validator("prompt_template")
    @classmethod
    def is_prompt_template(cls, v: PromptTemplate | dict) -> PromptTemplate:
        if isinstance(v, dict):
            return PromptTemplate(**v)
        elif isinstance(v, PromptTemplate):
            return v
        else:
            raise ValueError("Invalid value for prompt_template")

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
    sources: Optional[list[ChatSource]]


class SummaryBase(PersistableModel):
    file_uuids: list[UUID]
    file_hash: int


class Summary(SummaryBase):
    tasks: list[SummaryTask]


class SummaryComplete(SummaryBase):
    tasks: list[SummaryTaskComplete]
