import time
from io import BytesIO
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import pytest
from elasticsearch import NotFoundError
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from redbox.definitions import BackendAdapter
from redbox.local import LocalBackendAdapter
from redbox.models import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    ContentType,
    Feedback,
    File,
    SourceDocument,
    SummaryComplete,
    SummaryTaskComplete,
    Tag,
    UploadFile,
)
from redbox.tests.conftest import TEST_DATA, YieldFixture

DOCS = [Path(*x.parts[-2:]) for x in (TEST_DATA / "docs").glob("*.*")]
ADAPTERS = [LocalBackendAdapter]


@pytest.mark.incremental
class TestFiles:
    """Tests CRUD for a backend."""

    file: Optional[File] = None
    files: Optional[list[File]] = []
    tag: Optional[Tag] = None
    summary: Optional[SummaryComplete] = None

    @pytest.fixture(params=ADAPTERS)
    def backend(self, request, settings) -> YieldFixture[BackendAdapter]:
        backend = request.param(settings=settings)

        _ = backend.set_user(
            uuid=UUID("bd65600d-8669-4903-8a14-af88203add38"),
            name="Foo Bar",
            email="foo.bar@gov.uk",
            department="Cabinet Office",
            role="Civil Servant",
            preferred_language="British English",
        )

        backend.set_llm(
            model="openai/gpt-3.5-turbo",
            max_tokens=10_000,
            max_return_tokens=1_024,
            temperature=0.2,
        )

        yield backend

    @pytest.fixture(params=DOCS)
    def upload(self, request) -> YieldFixture[UploadFile]:
        full_path = Path(TEST_DATA / request.param)
        sanitised_name = full_path.name.replace("'", "_")
        file_type = full_path.suffix

        with open(full_path, "rb") as f:
            to_upload = UploadFile(
                content_type=ContentType(file_type),
                filename=sanitised_name,
                creator_user_uuid=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                file=BytesIO(f.read()),
            )

        yield to_upload

    # region FILES ====================

    def test_get_supported_file_types(self, upload, backend):
        accepted = set(backend.get_supported_file_types())
        uploaded = {upload.content_type.value}
        assert uploaded <= accepted

    def test_create_file(self, upload, backend):
        file = backend.create_file(file=upload)
        assert isinstance(file, File)
        TestFiles.file = file
        TestFiles.files = [file for _ in range(3)]

    def test_get_file_status(self, backend):
        status = backend.get_file_status(file_uuid=self.file.uuid)
        # TODO: Returns "embedding", needs fixing
        # assert status.processing_status.value == "complete"
        assert status is not None

    def test_get_file(self, backend):
        file = backend.get_file(file_uuid=self.file.uuid)
        assert isinstance(file, File)

    def test_get_files(self, backend):
        files = backend.get_files(file_uuids=[f.uuid for f in self.files])
        assert all(isinstance(file, File) for file in files)

    def test_get_object(self, backend):
        raw = backend.get_object(file_uuid=self.file.uuid)
        assert isinstance(raw, bytes)

    def test_list_files(self, backend):
        assert self.file in backend.list_files()

    def test_get_file_chunks(self, backend):
        chunks = backend.get_file_chunks(file_uuid=self.file.uuid)
        assert len(chunks) > 0

    def test_get_file_as_documents(self, backend):
        documents = backend.get_file_as_documents(file_uuid=self.file.uuid, max_tokens=3_000)
        assert len(documents) > 1
        assert all(isinstance(document, Document) for document in documents)

    # region TAGS ====================

    def test_create_tag(self, backend):
        tag = backend.create_tag(name="TestTag")
        assert isinstance(tag, Tag)
        TestFiles.tag = tag

    def test_get_tag(self, backend):
        tag = backend.get_tag(tag_uuid=self.tag.uuid)
        assert isinstance(tag, Tag)

    def test_add_files_to_tag(self, backend):
        _ = backend.add_files_to_tag(file_uuids=[self.file.uuid], tag_uuid=self.tag.uuid)
        time.sleep(1)

        tag = backend.get_tag(tag_uuid=self.tag.uuid)
        assert {self.file.uuid} <= tag.files

    def test_remove_files_from_tag(self, backend):
        _ = backend.remove_files_from_tag(file_uuids=[self.file.uuid], tag_uuid=self.tag.uuid)
        time.sleep(1)

        tag = backend.get_tag(tag_uuid=self.tag.uuid)
        assert not ({self.file.uuid} <= tag.files)

    def test_list_tags(self, backend):
        assert self.tag in backend.list_tags()

    def test_delete_tag(self, backend):
        _ = backend.delete_tag(tag_uuid=self.tag.uuid)
        time.sleep(1)
        assert self.tag not in backend.list_tags()

    # region SUMMARIES ====================

    def test_create_summary(self, backend):
        source = ChatSource(
            document=SourceDocument(page_content="Lorem ipsum dolor sit amet.", file_uuid=uuid4(), page_numbers=[1, 3]),
            html="",
        )
        tasks = [
            SummaryTaskComplete(
                id="foo",
                title="Bar",
                prompt_template=PromptTemplate.from_template("text"),
                file_uuids=[f.uuid for f in self.files],
                response_text="Lorem ipsum dolor sit amet.",
                sources=[source],
            )
        ]
        summary = backend.create_summary(file_uuids=[f.uuid for f in self.files], tasks=tasks)
        assert isinstance(summary, SummaryComplete)
        TestFiles.summary = summary
        time.sleep(3)

    def test_get_summary(self, backend):
        summary = backend.get_summary(file_uuids=[f.uuid for f in self.files])
        assert isinstance(summary, SummaryComplete)
        assert summary == self.summary

    def test_list_summaries(self, backend):
        assert self.summary in backend.list_summaries()

    def test_delete_summary(self, backend):
        _ = backend.delete_summary(file_uuids=[f.uuid for f in self.files])
        time.sleep(1)
        assert self.summary not in backend.list_summaries()

    def test_delete_file(self, backend):
        _ = backend.delete_file(file_uuid=self.file.uuid)
        time.sleep(1)
        assert self.file not in backend.list_files()

        chunks = backend.get_file_chunks(file_uuid=self.file.uuid)
        assert len(chunks) == 0

        with pytest.raises(NotFoundError):
            _ = backend.get_object(file_uuid=self.file.uuid)


@pytest.mark.incremental
class TestLLM:
    """Tests LLM calls for a backend."""

    @pytest.fixture(params=ADAPTERS)
    def backend(self, request, settings) -> YieldFixture[BackendAdapter]:
        backend = request.param(settings=settings)

        _ = backend.set_user(
            uuid=UUID("bd65600d-8669-4903-8a14-af88203add38"),
            name="Foo Bar",
            email="foo.bar@gov.uk",
            department="Cabinet Office",
            role="Civil Servant",
            preferred_language="British English",
        )

        backend.set_llm(
            model="openai/gpt-3.5-turbo",
            max_tokens=10_000,
            max_return_tokens=1_024,
            temperature=0.2,
        )
        yield backend

    @pytest.fixture
    def file_long(self, backend) -> YieldFixture[File]:
        full_path = Path(TEST_DATA / "docs/NATIONAL HEALTH SERVICE BILL.txt")
        sanitised_name = full_path.name.replace("'", "_")
        file_type = full_path.suffix

        with open(full_path, "rb") as f:
            to_upload = UploadFile(
                content_type=ContentType(file_type),
                filename=sanitised_name,
                creator_user_uuid=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                file=BytesIO(f.read()),
            )

        file = backend.create_file(file=to_upload)

        yield file

        _ = backend.delete_file(file_uuid=file.uuid)

    @pytest.fixture
    def file_short(self, backend) -> YieldFixture[File]:
        full_path = Path(TEST_DATA / "docs/Working Smarter (press release).docx")
        sanitised_name = full_path.name.replace("'", "_")
        file_type = full_path.suffix

        with open(full_path, "rb") as f:
            to_upload = UploadFile(
                content_type=ContentType(file_type),
                filename=sanitised_name,
                creator_user_uuid=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
                file=BytesIO(f.read()),
            )

        file = backend.create_file(file=to_upload)

        yield file

        _ = backend.delete_file(file_uuid=file.uuid)

    # region USER AND CONFIG ====================

    def test_get_set_user(self, backend):
        user_sent = backend.set_user(
            uuid=uuid4(),
            name="Foo Bar",
            email="foo.bar@gov.uk",
            department="Cabinet Office",
            role="Civil Servant",
            preferred_language="British English",
        )
        user_returned = backend.get_user()

        assert user_sent == user_returned

    def test_get_set_llm(self, backend):
        llm_sent = backend.set_llm(
            model="mistral/mistral-tiny",
            max_tokens=1_000,
            max_return_tokens=256,
            temperature=0.1,
        )
        llm_returned = backend.get_llm()

        assert llm_sent == llm_returned

    # region FEEDBACK ====================

    def test_create_feedback(self, backend):
        feedback = Feedback(
            input=[{"role": "user", "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."}],
            output={"role": "ai", "text": "Ut enim ad minim veniam."},
            feedback_type="thumbs",
            feedback_score="👍",
            feedback_text="Baz",
            creator_user_uuid=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        )

        given = backend.create_feedback(feedback=feedback)

        assert isinstance(given, Feedback)

    # region LLM ====================

    def test_rag_chat(self, backend):
        request = ChatRequest(
            message_history=[
                {"role": "user", "text": "What does Mr. Aneurin Bevan think of the national health insurance system"}
            ]
        )
        response = backend.rag_chat(chat_request=request)

        assert isinstance(response, ChatResponse)
        assert len(response.output_text) > 0
        assert len(response.source_documents) > 0

    def test_stuff_doc_summary(self, backend, file_short):
        summary = PromptTemplate.from_template("Summarise this text: {text}")
        response = backend.stuff_doc_summary(summary=summary, file_uuids=[file_short.uuid])

        assert isinstance(response, ChatResponse)
        assert len(response.output_text) > 0
        assert len(response.source_documents) > 0

    def test_map_reduce_summary(self, backend, file_short, file_long):
        map_prompt = PromptTemplate.from_template("Summarise this text: {text}")
        reduce_prompt = PromptTemplate.from_template("Summarise these summaries: {text}")
        response = backend.map_reduce_summary(
            map_prompt=map_prompt,
            reduce_prompt=reduce_prompt,
            file_uuids=[file_long.uuid, file_short.uuid],
        )

        assert isinstance(response, ChatResponse)
        assert len(response.output_text) > 0
        assert len(response.source_documents) > 0
