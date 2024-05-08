from typing import Literal, Optional

from pydantic import BaseModel, Field

from redbox.models.file import SourceDocument


class ChatMessage(BaseModel):
    text: str = Field(description="The text of the message")
    role: Literal["user", "ai", "system"] = Field(description="The role of the message")


class ChatSource(BaseModel):
    document: SourceDocument = Field(description="The source document")
    html: str = Field(description="The formatted HTML to display the document")


class ChatMessageSourced(ChatMessage):
    sources: list[ChatSource] = Field(description="The source documents")


class ChatRequest(BaseModel):
    message_history: list[ChatMessage] = Field(description="The history of messages in the chat")


class ChatResponse(BaseModel):
    source_documents: Optional[list[SourceDocument]] = Field(
        description="documents retrieved to form this response", default=None
    )
    output_text: str = Field(
        description="response text",
        examples=["The current Prime Minister of the UK is The Rt Hon. Rishi Sunak MP."],
    )
