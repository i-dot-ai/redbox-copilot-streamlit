from uuid import UUID
from typing import TextIO
import logging

from langchain_community.chat_models import ChatLiteLLM
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.elasticsearch import ApproxRetrievalStrategy, ElasticsearchStore

from redbox.definitions import BackendAdapter
from redbox.storage.elasticsearch import ElasticsearchStorageHandler
from redbox.parsing.file_chunker import FileChunker
from redbox.model_db import SentenceTransformerDB
from redbox.models import File, Settings, Chunk, FileStatus
from redbox.llm.llm_base import LLMHandler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class LocalBackendAdapter(BackendAdapter):
    def __init__(self, settings: Settings):
        # Settings
        self.settings = settings

        # Storage
        self.es = settings.elasticsearch_client()
        self.storage_handler = ElasticsearchStorageHandler(es_client=self.es, root_index="redbox-data")
        self.file_publisher = FileChunker(embedding_model=SentenceTransformerDB(settings.embedding_model))
        self.s3 = settings.s3_client()

        # LLM
        self.llm: ChatLiteLLM = None
        self.llm_handler: LLMHandler = None

    def _set_llm(self, model: str, max_tokens: int, temperature: int, user_uuid: UUID) -> None:
        self.llm = ChatLiteLLM(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            streaming=True,
        )  # type: ignore[call-arg]
        # A meta, private argument hasn't been typed properly in LangChain

        embedding_function = SentenceTransformerEmbeddings()

        hybrid = False
        if self.settings.elastic.subscription_level in ("platinum", "enterprise"):
            hybrid = True

        vector_store = ElasticsearchStore(
            index_name="redbox-vector",
            es_connection=self.es,
            embedding=embedding_function,
            strategy=ApproxRetrievalStrategy(hybrid=hybrid),
        )

        self.llm_handler = LLMHandler(
            llm=self.llm,
            user_uuid=user_uuid,
            vector_store=vector_store,
        )

    def add_file(self, file: File) -> UUID:
        self.storage_handler.write_item(file)

        log.info(f"publishing {file.uuid}")

        self.file_publisher.chunk_file(file)

        return file.uuid

    def get_file(self, file_uuid: UUID) -> File:
        return self.storage_handler.read_item(file_uuid, model_type="File")

    def delete_file(self, file_uuid: UUID) -> File:
        file = self.storage_handler.read_item(file_uuid, model_type="File")
        chunks = self.storage_handler.get_file_chunks(file.uuid)

        self.s3.delete_object(Bucket=self.settings.bucket_name, Key=file.key)
        self.storage_handler.delete_item(file)
        self.storage_handler.delete_items(chunks)

        return file

    def get_file_chunks(self, file_uuid: UUID) -> list[Chunk]:
        log.info(f"getting chunks for file {file_uuid}")
        return self.storage_handler.get_file_chunks(file_uuid)

    def get_file_status(self, file_uuid: UUID) -> FileStatus:
        status = self.storage_handler.get_file_status(file_uuid)
        return status

    def simple_chat(self, chat_history: list[dict]) -> TextIO:
        pass
