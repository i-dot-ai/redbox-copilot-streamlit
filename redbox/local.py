from uuid import UUID
from typing import Iterable, Sequence, Optional
import logging
from pathlib import Path
from datetime import date

from langchain_community.chat_models import ChatLiteLLM
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings

from redbox.definitions import BackendAdapter
from redbox.storage.elasticsearch import ElasticsearchStorageHandler
from redbox.parsing.file_chunker import FileChunker
from redbox.models import (
    File,
    Settings,
    Chunk,
    FileStatus,
    Tag,
    ChatMessage,
    UploadFile,
    ContentType,
    Summary,
    User,
    Feedback,
)
from redbox.llm.llm_base import LLMHandler
from redbox.llm.prompts.chat import get_chat_runnable

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class LocalBackendAdapter(BackendAdapter):
    def __init__(self, settings: Settings):
        # Settings
        self._settings: Settings = settings

        # Storage
        self._es = self._settings.elasticsearch_client()
        self._storage_handler = ElasticsearchStorageHandler(es_client=self._es, root_index="redbox-data")
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model,
            model_kwargs={"device": "cpu"},
            cache_folder=self._settings.sentence_transformers_home,
        )
        self._file_publisher = FileChunker(embedding_model=self._embedding_model)
        self._s3 = self._settings.s3_client()

        # LLM
        self._llm: Optional[ChatLiteLLM] = None
        self._llm_handler: Optional[LLMHandler] = None

        # User
        self._user: Optional[User] = None

    # ==================== USER AND CONFIG ====================
    def status(self):
        return {
            "llm": self._llm is not None,
            "llm_handler": self._llm_handler is not None,
            "user": self._user is not None,
            "file_publisher": self._file_publisher is not None,
            "elastic_client": self._es is not None,
            "storage_handler": self._storage_handler is not None,
            "embedding_model": self._embedding_model is not None,
            "s3": self._s3 is not None,
        }

    def set_user(self, name: str, email: str, uuid: UUID, department: str, role: str, preferred_language: str) -> None:
        self._user = User(
            name=name, email=email, uuid=uuid, department=department, role=role, preferred_language=preferred_language
        )

    def get_user(self) -> User:
        return self._user

    # ==================== FILES ====================
    def create_file(self, file: UploadFile) -> File:
        assert self._llm_handler is not None
        assert self._llm is not None
        assert self._user is not None

        # Upload
        log.info(f"Uploading {file.uuid}")

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
        simple_s3_url = authenticated_s3_url.split("?")[0]

        log.info(f"Uploaded file to {simple_s3_url}")

        file_uploaded = File(
            url=simple_s3_url,
            content_type=ContentType(file_type),
            name=file.filename,
            creator_user_uuid=file.creator_user_uuid,
        )

        # Chunk
        log.info(f"Chunking {file.uuid}")

        chunks = self._file_publisher.chunk_file(file_uploaded)

        # Save
        log.info(f"Saving {file.uuid}")

        self._storage_handler.write_item(file_uploaded)
        self._storage_handler.write_items(chunks)

        # Index
        log.info(f"Indexing {file.uuid}")

        self._llm_handler.add_chunks_to_vector_store(chunks=chunks)

        log.info(f"{file.uuid} complete!")

        return file_uploaded

    def get_file(self, file_uuid: UUID) -> File:
        return self._storage_handler.read_item(file_uuid, model_type="File")

    def get_files(self, file_uuids: list[UUID]) -> list[File]:
        return self._storage_handler.read_items(file_uuids, model_type="File")

    def get_object(self, file_uuid: UUID) -> bytes:
        file = self.get_file(file_uuid=file_uuid)
        file_object = self._s3.get_object(Bucket=self._settings.bucket_name, Key=file.name)
        return file_object["Body"].read()

    def list_files(self) -> Sequence[File]:
        files = self._storage_handler.read_all_items(model_type="File")
        assert all(isinstance(file, File) for file in files)

        return files

    def delete_file(self, file_uuid: UUID) -> File:
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
        log.info(f"getting chunks for file {file_uuid}")
        chunks = self._storage_handler.get_file_chunks(file_uuid)
        return chunks

    def get_file_status(self, file_uuid: UUID) -> FileStatus:
        status = self._storage_handler.get_file_status(file_uuid)
        return status

    def get_supported_file_types(self) -> list[str]:
        return self._file_publisher.supported_file_types

    # ==================== FEEDBACK ====================
    def create_feedback(self, feedback: Feedback) -> Feedback:
        self._storage_handler.write_item(feedback)
        return feedback

    # ==================== TAGS ====================
    def create_tag(self, name: str) -> Tag:
        assert self._user is not None

        tag = Tag(name=name, files=set(), creator_user_uuid=self._user.uuid)
        self._storage_handler.write_item(item=tag)
        return tag

    def get_tag(self, tag_uuid: UUID) -> Tag:
        return self._storage_handler.read_item(item_uuid=tag_uuid, model_type="Tag")

    def list_tags(self) -> Sequence[Tag]:
        tags = self._storage_handler.read_all_items(model_type="Tag")
        assert all(isinstance(tag, Tag) for tag in tags)

        return tags

    def delete_tag(self, tag_uuid: UUID) -> Tag:
        tag = self.get_tag(tag_uuid=tag_uuid)
        self._storage_handler.delete_item(item=tag)
        return tag

    def add_files_to_tag(self, file_uuids: Sequence[UUID], tag_uuid: UUID) -> Tag:
        tag = self.get_tag(tag_uuid=tag_uuid)
        tag.files.update(file_uuids)
        self._storage_handler.update_item(item=tag)
        return tag

    def remove_files_from_tag(self, file_uuids: Sequence[UUID], tag_uuid: UUID) -> Tag:
        tag = self.get_tag(tag_uuid=tag_uuid)
        for file_uuid in file_uuids:
            tag.files.discard(file_uuid)
        self._storage_handler.update_item(item=tag)
        return tag

    # ==================== SUMMARY ====================
    def create_summary(self, summary: Summary) -> Summary:
        pass

    def get_summary(self, summary_uuid: UUID) -> Summary:
        pass

    def delete_summary(self, summary_uuid: UUID) -> Summary:
        pass

    def list_summaries(self) -> Sequence[Summary]:
        pass

    # ==================== LLM ====================
    def set_llm(self, model: str, max_tokens: int, temperature: int) -> None:
        self._llm = ChatLiteLLM(
            model=model,
            max_tokens=max_tokens,
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
            embedding=self._embedding_model,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=hybrid),
        )

        self._llm_handler = LLMHandler(
            llm=self._llm,
            vector_store=vector_store,
        )

    def set_chat_prompt(self, init_messages: list[ChatMessage]) -> None:
        self._llm_handler._chat_runnable = get_chat_runnable(
            llm=self._llm_handler._llm, get_history_func=self._llm_handler.get_chat, init_messages=init_messages
        )

    def get_chat(self, chat_uuid: UUID) -> BaseChatMessageHistory:
        return self._llm_handler.get_chat(chat_uuid=chat_uuid)

    def simple_chat(self, input: str, chat_uuid: UUID) -> str:
        return self._llm_handler.chat(input=input, chat_uuid=chat_uuid)

    def simple_chat_stream(self, input: str, chat_uuid: UUID) -> Iterable:
        return self._llm_handler.chat_stream(input=input, chat_uuid=chat_uuid)

    def rag_chat(self, question: str, chat_uuid: UUID) -> str:
        response = self._llm_handler.chat_with_rag(
            question=question,
            chat_uuid=chat_uuid,
            current_date=date.today().isoformat(),
            user_info=self.get_user().summary(),
        )
        return response

    def rag_chat_stream(self, question: str, chat_uuid: UUID) -> Iterable:
        response = self._llm_handler.chat_with_rag_stream(
            question=question,
            chat_uuid=chat_uuid,
            current_date=date.today().isoformat(),
            user_info=self.get_user().summary(),
        )
        return response
