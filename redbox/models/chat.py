from typing import Literal, Optional

from pydantic import Field, BaseModel, field_validator, field_serializer

from langchain_core.documents.base import Document


class ChatMessage(BaseModel):
    text: str = Field(description="The text of the message")
    role: Literal["user", "ai", "system"] = Field(description="The role of the message")


class ChatSource(BaseModel):
    document: object = Field(default=None, description="The source document")
    html: str = Field(default=None, description="The formatted HTML to display the document")

    @field_validator("document")
    @classmethod
    def is_document(cls, v: Document) -> list[Document]:
        """Custom validator for using Pydantic v1 object in model."""
        assert isinstance(v, Document)
        return v

    @field_serializer("document")
    def serialise_doc(self, document: Document, _info):
        return document.dict()


class ChatMessageSourced(ChatMessage):
    sources: list[ChatSource] = Field(default=None, description="The source documents")


class ChatRequest(BaseModel):
    message_history: list[ChatMessage] = Field(description="The history of messages in the chat")


class ChatResponse(BaseModel):
    response_message: ChatMessage = Field(description="The response message")
    sources: Optional[list[object]] = Field(default=None, description="The source documents")

    @field_validator("sources")
    @classmethod
    def is_document(cls, v: list[Document]) -> list[Document]:
        """Custom validator for using Pydantic v1 object in model."""
        assert all(isinstance(doc, Document) for doc in v)
        return v
