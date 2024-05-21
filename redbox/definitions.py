from abc import ABC, abstractmethod
from typing import Callable, Literal, Optional, Sequence
from uuid import UUID

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document

from redbox.llm.llm_base import LLMHandler
from redbox.models import (
    ChatRequest,
    ChatResponse,
    Chunk,
    Feedback,
    File,
    FileStatus,
    SummaryComplete,
    SummaryTaskComplete,
    Tag,
    UploadFile,
    User,
)


class Backend(ABC):
    @property
    @abstractmethod
    def status(self) -> dict[str, bool]:
        """Reports the current state of set variables."""
        ...

    @abstractmethod
    def health(self) -> Literal["ready"]:
        """Returns the health of the API."""
        ...

    @abstractmethod
    def set_user(
        self,
        name: str,
        email: str,
        uuid: UUID,
        department: str,
        role: str,
        preferred_language: str,
    ) -> User:
        """Sets the user attribute."""
        ...

    @abstractmethod
    def get_user(self) -> User:
        """Gets the user attribute."""
        ...

    # region FILES ====================

    @abstractmethod
    def create_file(self, file: UploadFile) -> File:
        """Creates, chunks and embeds a file."""
        ...

    @abstractmethod
    def get_file(self, file_uuid: UUID) -> File:
        """Gets a file object by UUID."""
        ...

    @abstractmethod
    def get_files(self, file_uuids: list[UUID]) -> list[File]:
        """Gets many file objects by UUID."""
        ...

    @abstractmethod
    def get_object(self, file_uuid: UUID) -> bytes:
        """Gets a raw file blob by UUID."""
        ...

    @abstractmethod
    def list_files(self) -> Sequence[File]:
        """Lists all file objects in the system."""
        ...

    @abstractmethod
    def delete_file(self, file_uuid: UUID) -> File:
        """Deletes a file object by UUID."""
        ...

    @abstractmethod
    def get_file_chunks(self, file_uuid: UUID) -> Sequence[Chunk]:
        """Gets a file's chunks by UUID."""
        ...

    @abstractmethod
    def get_file_as_documents(self, file_uuid: UUID, max_tokens: int) -> Sequence[Document]:
        """Gets a file as LangChain Documents, splitting it by max_tokens."""
        ...

    @abstractmethod
    def get_file_token_count(self, file_uuid: UUID) -> int:
        """Gets a file's token count."""
        ...

    @abstractmethod
    def get_file_status(self, file_uuid: UUID) -> FileStatus:
        """Gets the processing status of a file."""
        ...

    @abstractmethod
    def get_supported_file_types(self) -> list[str]:
        """Shows the filetypes the system can process."""
        ...

    # region FEEDBACK ====================

    @abstractmethod
    def create_feedback(self, feedback: Feedback) -> Feedback:
        """Records a feedback object in the system."""
        ...

    # region TAGS ====================

    @abstractmethod
    def create_tag(self, name: str) -> Tag:
        """Creates a new tag with the given name."""
        ...

    @abstractmethod
    def get_tag(self, tag_uuid: UUID) -> Tag:
        """Gets a tag object by UUID."""
        ...

    @abstractmethod
    def add_files_to_tag(self, file_uuids: list[UUID], tag_uuid: UUID) -> Tag:
        """Adds files to the specified tag by UUID."""
        ...

    @abstractmethod
    def remove_files_from_tag(self, file_uuids: list[UUID], tag_uuid: UUID) -> Tag:
        """Removes files from the specified tag by UUID."""
        ...

    @abstractmethod
    def list_tags(self) -> Sequence[Tag]:
        """Lists all tag objects in the system."""
        ...

    @abstractmethod
    def delete_tag(self, tag_uuid: UUID) -> Tag:
        """Deletes a tag object by UUID."""
        ...

    # region SUMMARIES ====================

    @abstractmethod
    def create_summary(self, file_uuids: list[UUID], tasks: list[SummaryTaskComplete]) -> SummaryComplete:
        """Creates a new summary with the given files and tasks."""
        ...

    @abstractmethod
    def get_summary(self, file_uuids: list[UUID]) -> SummaryComplete | None:
        """Gets a summary object by UUID."""
        ...

    @abstractmethod
    def list_summaries(self) -> Sequence[SummaryComplete]:
        """Lists all summary objects in the system."""
        ...

    @abstractmethod
    def delete_summary(self, file_uuids: list[UUID]) -> SummaryComplete:
        """Deletes a summary object by UUID."""
        ...

    # region LLM ====================

    @abstractmethod
    def set_llm(self, model: str, max_tokens: int, max_return_tokens: int, temperature: float) -> LLMHandler:
        """Sets the LLM the backend will use for all operations.

        Args:
            model (str): A LiteLLM-compatible string of the model name
            max_tokens (int): The size to limit the LLM's context window to
            max_return_tokens (int): The max tokens the LLM with respond with
            temperature (float): 0-1. How creative the LLM will be. 1 is high.
        """
        ...

    @abstractmethod
    def get_llm(self) -> LLMHandler:
        """Gets the LLM currently in use."""
        ...

    @abstractmethod
    def rag_chat(self, chat_request: ChatRequest, callbacks: Optional[list[Callable]] = None) -> ChatResponse:
        """Given a chat history, have the LLM respond with reference to files in the box."""
        ...

    @abstractmethod
    def stuff_doc_summary(
        self, summary: PromptTemplate, file_uuids: list[UUID], callbacks: Optional[list[Callable]] = None
    ) -> ChatResponse:
        """Given a task and some files, have the LLM summarise those files via stuffing.

        Will put the contents of the files directly into the LLM's context window. Will
        fail if more tokens are needed than the LLM can cope with.
        """
        ...

    @abstractmethod
    def map_reduce_summary(
        self,
        map_prompt: PromptTemplate,
        reduce_prompt: PromptTemplate,
        file_uuids: list[UUID],
        callbacks: Optional[list[Callable]] = None,
    ) -> ChatResponse:
        """Given a task and some files, have the LLM summarise those files via map reduce.

        Will first summarise the documents one by one with the map prompt, then summarise
        those summaries with the reduce prompt.
        """
        ...
