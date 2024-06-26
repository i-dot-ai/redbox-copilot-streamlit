from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    model: str
    max_seq_length: int
    vector_size: int


class Embedding(BaseModel):
    object: Literal["embedding"]
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"]
    data: list[Embedding]
    embedding_id: str
    embedding_model: str
    embedding_model_info: ModelInfo


class EmbeddingRequest(BaseModel):
    sentences: list[str]


class StatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str


class EmbedQueueItem(BaseModel):
    """Instruction to Ingest app for what to embed"""

    chunk_uuid: UUID = Field(description="id of the chunk that this text belongs to")
