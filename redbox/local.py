import logging
import urllib.parse
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Sequence
from uuid import UUID

from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore

from redbox.definitions import BackendAdapter
from redbox.llm.llm_base import LLMHandler
from redbox.models import (
    ChatRequest,
    ChatResponse,
    Chunk,
    SourceDocument,
    Feedback,
    User,
    UploadFile,
    ContentType,
    Feedback,
    File,
    FileStatus,
    Metadata,
    Settings,
    SourceDocument,
    SummaryComplete,
    SummaryTaskComplete,
    Tag,
    UploadFile,
    User,
)
from redbox.parsing.file_chunker import FileChunker
from redbox.storage.elasticsearch import ElasticsearchStorageHandler


class LocalBackendAdapter(BackendAdapter):
    def __init__(self, settings: Settings):
        # Settings
        self._settings: Settings = settings

        # Storage
        self._es = self._settings.elasticsearch_client()
        self._storage_handler = ElasticsearchStorageHandler(es_client=self._es, root_index="redbox-data")
        self._s3 = self._settings.s3_client()
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model,
            model_kwargs={"device": "cpu"},
            cache_folder=self._settings.sentence_transformers_home,
        )
        self._file_publisher = FileChunker(
            embedding_model=self._embedding_model, s3_client=self._s3, bucket_name=self._settings.bucket_name
        )

        # LLM
        self._llm: Optional[LLMHandler] = None

        # User
        self._user: Optional[User] = None

    # region USER AND CONFIG ====================

    @property
    def status(self) -> dict[str, bool]:
        """Reports the current state of set variables."""
        return {
            "llm": self._llm is not None,
            "user": self._user is not None,
            "file_publisher": self._file_publisher is not None,
            "elastic_client": self._es is not None,
            "storage_handler": self._storage_handler is not None,
            "embedding_model": self._embedding_model is not None,
            "s3": self._s3 is not None,
        }

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
        self._user = User(
            name=name,
            email=email,
            uuid=uuid,
            department=department,
            role=role,
            preferred_language=preferred_language,
        )
        return self._user

    def get_user(self):
        """Gets the user attribute."""
        return self._user

    # region FILES ====================

    def create_file(self, file: UploadFile) -> File:
        """Creates, chunks and embeds a file."""
        assert self._llm is not None
        assert self._user is not None

        # Upload
        logging.info(f"Uploading {file.uuid}")

        file_type = Path(file.filename).suffix

        self._s3.put_object(
            Bucket=self._settings.bucket_name,
            Body=file.file,
            Key=file.filename,
            Tagging=f"file_type={file_type}&user_uuid={file.creator_user_uuid}",
        )

        authenticated_s3_url = self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._settings.bucket_name, "Key": file.filename},
            ExpiresIn=3600,
        )

        # Strip off the query string (we don't need the keys)
        authenticated_s3_url_parsed = urllib.parse.urlparse(authenticated_s3_url)
        simple_s3_url = urllib.parse.urljoin(authenticated_s3_url, authenticated_s3_url_parsed.path)

        logging.info(f"Uploaded file to {simple_s3_url}")

        file_uploaded = File(
            url=simple_s3_url,
            content_type=ContentType(file_type),
            name=file.filename,
            creator_user_uuid=file.creator_user_uuid,
        )

        # Chunk
        logging.info(f"Chunking {file.uuid}")

        chunks = self._file_publisher.chunk_file(file_uploaded)

        # Save
        logging.info(f"Saving {file.uuid}")

        self._storage_handler.write_item(file_uploaded)
        self._storage_handler.write_items(chunks)

        # Index
        logging.info(f"Indexing {file.uuid}")

        self._llm.add_chunks_to_vector_store(chunks=chunks)

        logging.info(f"{file.uuid} complete!")

        return file_uploaded

    def get_file(self, file_uuid: UUID) -> File:
        """Gets a file object by UUID."""
        return self._storage_handler.read_item(file_uuid, model_type="File")

    def get_files(self, file_uuids: list[UUID]) -> list[File]:
        """Gets many file objects by UUID."""
        return self._storage_handler.read_items(file_uuids, model_type="File")

    def get_object(self, file_uuid: UUID) -> bytes:
        """Gets a raw file blob by UUID."""
        file = self.get_file(file_uuid=file_uuid)
        file_object = self._s3.get_object(Bucket=self._settings.bucket_name, Key=file.name)
        return file_object["Body"].read()

    def list_files(self) -> Sequence[File]:
        """Lists all file objects in the system."""
        files = self._storage_handler.read_all_items(model_type="File")
        assert all(isinstance(file, File) for file in files)

        return files

    def delete_file(self, file_uuid: UUID) -> File:
        """Deletes a file object by UUID."""
        file = self.get_file(file_uuid=file_uuid)
        chunks = self._storage_handler.get_file_chunks(file.uuid)

        self._s3.delete_object(Bucket=self._settings.bucket_name, Key=file.name)
        self._storage_handler.delete_item(file)
        self._storage_handler.delete_items(chunks)

        for tag in self.list_tags():
            _ = self.remove_files_from_tag(file_uuids=[file.uuid], tag_uuid=tag.uuid)
            if len(tag.files) == 0:
                _ = self.delete_tag(tag_uuid=tag.uuid)

        return file

    def get_file_chunks(self, file_uuid: UUID) -> Sequence[Chunk]:
        """Gets a file's chunks by UUID."""
        logging.info(f"getting chunks for file {file_uuid}")
        chunks = self._storage_handler.get_file_chunks(file_uuid)
        return chunks

    def get_file_as_documents(self, file_uuid: UUID, max_tokens: int) -> Sequence[Document]:
        """Gets a file as LangChain Documents, splitting it by max_tokens."""
        documents: list[Document] = []
        chunks = self.get_file_chunks(file_uuid=file_uuid)

        token_count: int = 0
        page_content: list[str] = []
        metadata: list[Metadata] = []

        for chunk in chunks:
            if token_count + chunk.token_count >= max_tokens:
                document = Document(
                    page_content=" ".join(page_content),
                    metadata=reduce(Metadata.merge, metadata),
                )
                documents.append(document)
                token_count: int = 0
                page_content: list[str] = []
                metadata: list[Metadata] = []

            page_content.append(chunk.text)
            metadata.append(chunk.metadata)
            token_count += chunk.token_count

        if len(page_content) > 0:
            document = Document(
                page_content=" ".join(page_content),
                metadata=reduce(Metadata.merge, metadata),
            )
            documents.append(document)

        return documents

    def get_file_status(self, file_uuid: UUID) -> FileStatus:
        """Gets the processing status of a file."""
        status = self._storage_handler.get_file_status(file_uuid)
        return status

    def get_supported_file_types(self) -> list[str]:
        """Shows the filetypes the system can process."""
        return self._file_publisher.supported_file_types

    # region FEEDBACK ====================

    def create_feedback(self, feedback: Feedback) -> Feedback:
        """Records a feedback object in the system."""
        self._storage_handler.write_item(feedback)
        return feedback

    # region TAGS ====================

    def create_tag(self, name: str) -> Tag:
        """Creates a new tag with the given name."""
        assert self._user is not None

        tag = Tag(name=name, files=set(), creator_user_uuid=self.get_user().uuid)
        self._storage_handler.write_item(item=tag)
        return tag

    def get_tag(self, tag_uuid: UUID) -> Tag:
        """Gets a tag object by UUID."""
        return self._storage_handler.read_item(item_uuid=tag_uuid, model_type="Tag")

    def add_files_to_tag(self, file_uuids: list[UUID], tag_uuid: UUID) -> Tag:
        """Adds files to the specified tag by UUID."""
        tag = self.get_tag(tag_uuid=tag_uuid)
        tag.files.update(file_uuids)
        self._storage_handler.update_item(item=tag)
        return tag

    def remove_files_from_tag(self, file_uuids: list[UUID], tag_uuid: UUID) -> Tag:
        """Removes files from the specified tag by UUID."""
        tag = self.get_tag(tag_uuid=tag_uuid)
        for file_uuid in file_uuids:
            tag.files.discard(file_uuid)
        self._storage_handler.update_item(item=tag)
        return tag

    def list_tags(self) -> Sequence[Tag]:
        """Lists all tag objects in the system."""
        tags = self._storage_handler.read_all_items(model_type="Tag")
        assert all(isinstance(tag, Tag) for tag in tags)

        return tags

    def delete_tag(self, tag_uuid: UUID) -> Tag:
        """Deletes a tag object by UUID."""
        tag = self.get_tag(tag_uuid=tag_uuid)
        self._storage_handler.delete_item(item=tag)
        return tag

    # region SUMMARIES ====================

    def create_summary(self, file_uuids: list[UUID], tasks: list[SummaryTaskComplete]) -> SummaryComplete:
        """Creates a new summary with the given files and tasks."""
        assert self._user is not None

        summary = SummaryComplete(
            file_hash=hash(tuple(sorted(file_uuids))),
            file_uuids=file_uuids,
            tasks=tasks,
            creator_user_uuid=self.get_user().uuid,
        )

        self._storage_handler.write_item(item=summary)
        return summary

    def get_summary(self, file_uuids: list[UUID]) -> SummaryComplete | None:
        """Gets a summary object by UUID."""
        file_hash = hash(tuple(sorted(file_uuids)))

        for summary in self.list_summaries():
            if summary.file_hash == file_hash:
                return summary

        return None

    def list_summaries(self) -> Sequence[SummaryComplete]:
        """Lists all summary objects in the system."""
        summaries = self._storage_handler.read_all_items(model_type="SummaryComplete")
        assert all(isinstance(summary, SummaryComplete) for summary in summaries)

        return summaries

    def delete_summary(self, file_uuids: list[UUID]) -> SummaryComplete:
        """Deletes a summary object by UUID."""
        summary = self.get_summary(file_uuids=file_uuids)
        self._storage_handler.delete_item(item=summary)
        return summary

    # region LLM ====================

    def set_llm(self, model: str, max_tokens: int, max_return_tokens: int, temperature: float) -> LLMHandler:
        """Sets the LLM the backend will use for all operations.

        Args:
            model (str): A LiteLLM-compatible string of the model name
            max_tokens (int): The size to limit the LLM's context window to
            max_return_tokens (int): The max tokens the LLM with respond with
            temperature (float): 0-1. How creative the LLM will be. 1 is high.
        """
        llm = ChatLiteLLM(
            model=model,
            max_tokens=max_return_tokens,  # max count of returned tokens, NOT context size
            temperature=temperature,
            streaming=True,
        )  # type: ignore[call-arg]
        # A meta, private argument hasn't been typed properly in LangChain

        hybrid = False
        if self._settings.elastic.subscription_level in ("platinum", "enterprise"):
            hybrid = True

        vector_store = ElasticsearchStore(
            index_name="redbox-vector",
            es_connection=self._es,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=hybrid),
            embedding=self._embedding_model,
        )

        self._llm = LLMHandler(
            llm=llm, user_uuid=str(self.get_user().uuid), vector_store=vector_store, max_tokens=max_tokens
        )

        return self._llm

    def get_llm(self) -> LLMHandler:
        """Gets the LLM currently in use."""
        return self._llm

    def simple_chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Given a chat history, have the LLM respond."""
        pass

    def rag_chat(self, chat_request: ChatRequest, callbacks: Optional[list[Callable]] = None) -> ChatResponse:
        """Given a chat history, have the LLM respond with reference to files in the box."""
        *previous_history, question = chat_request.message_history

        formatted_history = "\n".join([f"{msg.role}: {msg.text}" for msg in previous_history])

        response = self._llm.chat_with_rag(
            user_question=question.text,
            user_info=self.get_user().dict_llm(),
            chat_history=formatted_history,
            callbacks=callbacks or [],
        )

        return ChatResponse(
            output_text=response["output_text"],
            source_documents=[
                SourceDocument.from_langchain_document(document=document) for document in response["input_documents"]
            ],
        )

    def stuff_doc_summary(
        self, summary: PromptTemplate, file_uuids: list[UUID], callbacks: Optional[list[Callable]] = None
    ) -> ChatResponse:
        """Given a task and some files, have the LLM summarise those files via stuffing.

        Will put the contents of the files directly into the LLM's context window. Will
        fail if more tokens are needed than the LLM can cope with.
        """
        documents: list[Document] = []
        for file_uuid in file_uuids:
            chunks = self.get_file_chunks(file_uuid=file_uuid)
            document = Document(
                page_content=" ".join(chunk.text for chunk in chunks),
                metadata=reduce(Metadata.merge, (chunk.metadata for chunk in chunks)),
            )
            documents.append(document)

        response = self._llm.stuff_doc_summary(
            prompt=summary,
            documents=documents,
            user_info=self.get_user().dict_llm(),
            callbacks=callbacks or [],
        )

        return ChatResponse(
            output_text=response,
            source_documents=[SourceDocument.from_langchain_document(document) for document in documents],
        )

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
        # Create documents under max_tokens size
        documents: list[Document] = []
        for file_uuid in file_uuids:
            file_as_documents = self.get_file_as_documents(file_uuid=file_uuid, max_tokens=self._llm.max_tokens)
            documents += file_as_documents

        response = self._llm.map_reduce_summary(
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt,
            documents=documents,
            user_info=self.get_user().dict_llm(),
            callbacks=callbacks or [],
        )

        return ChatResponse(
            output_text=response,
            source_documents=[SourceDocument.from_langchain_document(document) for document in documents],
        )
