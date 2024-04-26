from uuid import UUID
from typing import TextIO, Sequence, Optional
import logging
from pathlib import Path

from langchain_community.chat_models import ChatLiteLLM

from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from redbox.definitions import BackendAdapter
from redbox.models.file import UploadFile, ContentType
from redbox.storage.elasticsearch import ElasticsearchStorageHandler
from redbox.parsing.file_chunker import FileChunker
from redbox.models import File, Settings, Chunk, FileStatus, Tag
from redbox.llm.llm_base import LLMHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


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
        self._user_uuid: Optional[UUID] = None

    def _set_uuid(self, user_uuid: UUID) -> None:
        self._user_uuid = user_uuid

    def _set_llm(self, model: str, max_tokens: int, temperature: int) -> None:
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
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=hybrid),
            embedding=self._embedding_model,
        )

        self._llm_handler = LLMHandler(
            llm=self._llm,
            user_uuid=str(self._user_uuid),
            vector_store=vector_store,
        )

    def get_supported_file_types(self) -> list[str]:
        return self._file_publisher.supported_file_types

    def add_file(self, file: UploadFile) -> File:
        assert self._llm_handler is not None
        assert self._llm is not None
        assert self._user_uuid is not None

        # ==================== UPLOAD ====================
        log.info(f"uploading {file.uuid}")

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

        file_uploaded = File(
            url=simple_s3_url,
            content_type=ContentType(file_type),
            name=file.filename,
            creator_user_uuid=file.creator_user_uuid,
        )

        # ==================== CHUNKING ====================
        log.info(f"chunking {file.uuid}")

        chunks = self._file_publisher.chunk_file(file_uploaded)

        # ==================== SAVING ====================
        log.info(f"saving {file.uuid}")

        self._storage_handler.write_item(file_uploaded)
        self._storage_handler.write_items(chunks)

        # ==================== INDEXING ====================
        log.info(f"indexing {file.uuid}")

        self._llm_handler.add_chunks_to_vector_store(chunks=chunks)

        log.info(f"{file.uuid} complete!")

        return file_uploaded

    def get_file(self, file_uuid: UUID) -> File:
        return self._storage_handler.read_item(file_uuid, model_type="File")
    
    def get_files(self, file_uuids: list[UUID]) -> list[File]:
        return self._storage_handler.read_items(file_uuids, model_type="File")

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

    def list_tags(self) -> Sequence[Tag]:
        tags = self._storage_handler.read_all_items(model_type="Tag")
        assert all(isinstance(tag, Tag) for tag in tags)

        return tags

    def get_tag(self, tag_uuid: UUID) -> Tag:
        return self._storage_handler.read_item(item_uuid=tag_uuid, model_type="Tag")

    def add_files_to_tag(self, file_uuids: list[UUID], tag_uuid: UUID) -> Tag:
        tag = self.get_tag(tag_uuid=tag_uuid)
        tag.files.update(file_uuids)
        self._storage_handler.update_item(item=tag)
        return tag

    def remove_files_from_tag(self, file_uuids: list[UUID], tag_uuid: UUID) -> Tag:
        tag = self.get_tag(tag_uuid=tag_uuid)
        for file_uuid in file_uuids:
            tag.files.discard(file_uuid)
        self._storage_handler.update_item(item=tag)
        return tag

    def create_tag(self, name: str) -> Tag:
        assert self._user_uuid is not None

        tag = Tag(name=name, files=set(), creator_user_uuid=self._user_uuid)
        self._storage_handler.write_item(item=tag)
        return tag

    def delete_tag(self, tag_uuid: UUID) -> Tag:
        tag = self.get_tag(tag_uuid=tag_uuid)
        self._storage_handler.delete_item(item=tag)
        return tag

    def simple_chat(self, chat_history: Sequence[dict]) -> TextIO:
        pass
