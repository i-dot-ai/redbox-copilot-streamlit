from typing import TextIO, Sequence

from abc import ABC, abstractmethod

from uuid import UUID

from redbox.models import File, Chunk, FileStatus, UploadFile


class BackendAdapter(ABC):
    @abstractmethod
    def add_file(self, file: UploadFile) -> File: ...

    @abstractmethod
    def get_file(self, file_uuid: UUID) -> File: ...

    @abstractmethod
    def delete_file(self, file_uuid: UUID) -> File: ...

    @abstractmethod
    def get_file_chunks(self, file_uuid: UUID) -> Sequence[Chunk]: ...

    @abstractmethod
    def get_file_status(self, file_uuid: UUID) -> FileStatus: ...

    @abstractmethod
    def simple_chat(self, chat_history: Sequence[dict]) -> TextIO: ...