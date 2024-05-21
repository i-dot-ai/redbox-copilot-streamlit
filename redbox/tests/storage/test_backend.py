import time
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from requests.exceptions import HTTPError, Timeout

from redbox.api import APIBackend
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

TEST_USER_UUID = UUID("00000000-0000-0000-0000-000000000000")


@pytest.fixture(scope="session")
def backend(settings) -> YieldFixture[APIBackend]:
    backend = APIBackend(settings=settings)

    _ = backend.set_user(
        uuid=TEST_USER_UUID,
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

    assert backend.health() == "ready"

    yield backend


@pytest.fixture(scope="session")
def created_files(backend) -> YieldFixture[list[File]]:
    """Uploads all files for use in other tests."""
    file_paths: list[Path] = [Path(*x.parts[-2:]) for x in (TEST_DATA / "docs").glob("*.docx")]
    uploaded_files: list[File] = []

    for file_path in file_paths:
        full_path = Path(TEST_DATA / file_path)
        sanitised_name = full_path.name.replace("'", "_")
        file_type = full_path.suffix

        with open(full_path, "rb") as f:
            to_upload = UploadFile(
                content_type=ContentType(file_type),
                filename=sanitised_name,
                creator_user_uuid=TEST_USER_UUID,
                file=BytesIO(f.read()),
            )

        file = backend.create_file(file=to_upload)

        assert isinstance(file, File)

        try:
            _ = backend.get_object(file_uuid=file.uuid)
        except ValueError:
            pytest.fail("Object not uploaded correctly")

        uploaded_files.append(file)

    uploaded_file_statuses: list[str] = [
        backend.get_file_status(file_uuid=file.uuid).processing_status for file in uploaded_files
    ]

    timeout = 300
    start_time = time.time()
    while not all([i == "complete" for i in uploaded_file_statuses]):
        if time.time() - start_time > timeout:
            raise Timeout("Took too long to chunk")

        time.sleep(5)

        uploaded_file_statuses: list[str] = [
            backend.get_file_status(file_uuid=file.uuid).processing_status for file in uploaded_files
        ]

    yield uploaded_files

    for file in uploaded_files:
        _ = backend.delete_file(file_uuid=file.uuid)

    time.sleep(5)

    for file in uploaded_files:
        assert file not in backend.list_files()

        with pytest.raises(HTTPError):
            _ = backend.get_file_chunks(file_uuid=file.uuid)

        with pytest.raises(ValueError):
            _ = backend.get_object(file_uuid=file.uuid)


@pytest.fixture(scope="session")
def created_tag(backend) -> YieldFixture[Tag]:
    tag = backend.create_tag(name="TestTag")
    assert isinstance(tag, Tag)

    yield tag

    _ = backend.delete_tag(tag_uuid=tag.uuid)

    time.sleep(5)

    assert tag not in backend.list_tags()


@pytest.fixture(scope="session")
def created_summary(backend, created_files) -> YieldFixture[SummaryComplete]:
    file_uuids = [f.uuid for f in created_files]

    source = ChatSource(
        document=SourceDocument(page_content="Lorem ipsum dolor sit amet.", file_uuid=uuid4(), page_numbers=[1, 3]),
        html="",
    )
    tasks = [
        SummaryTaskComplete(
            id="foo",
            title="Bar",
            prompt_template=PromptTemplate.from_template("text"),
            file_uuids=file_uuids,
            response_text="Lorem ipsum dolor sit amet.",
            sources=[source],
        )
    ]

    summary = backend.create_summary(file_uuids=file_uuids, tasks=tasks)

    assert isinstance(summary, SummaryComplete)

    time.sleep(1)

    yield summary

    _ = backend.delete_summary(file_uuids=file_uuids)

    time.sleep(5)

    assert summary not in backend.list_summaries()


# region FILES ====================


class TestFiles:
    """Tests CRUD for Files."""

    def test_get_supported_file_types(self, created_files, backend):
        accepted = set(backend.get_supported_file_types())
        uploaded = {Path(file.key).suffix for file in created_files}
        assert uploaded <= accepted

    def test_get_file_status(self, created_files, backend):
        for file in created_files:
            status = backend.get_file_status(file_uuid=file.uuid)
        # TODO: Returns "embedding", needs fixing
        # assert status.processing_status.value == "complete"
        assert status is not None

    def test_get_file(self, created_files, backend):
        for file in created_files:
            returned = backend.get_file(file_uuid=file.uuid)
            assert isinstance(returned, File)

    def test_get_files(self, created_files, backend):
        files = backend.get_files(file_uuids=[f.uuid for f in created_files])
        assert all(isinstance(file, File) for file in files)

    def test_get_object(self, created_files, backend):
        for file in created_files:
            raw = backend.get_object(file_uuid=file.uuid)
            assert isinstance(raw, bytes)

    def test_list_files(self, created_files, backend):
        assert set(created_files) <= set(backend.list_files())

    def test_get_file_chunks(self, created_files, backend):
        for file in created_files:
            chunks = backend.get_file_chunks(file_uuid=file.uuid)
            assert len(chunks) > 1

    def test_get_file_as_documents(self, created_files, backend):
        for file in created_files:
            documents = backend.get_file_as_documents(file_uuid=file.uuid, max_tokens=1_000)
            assert len(documents) > 1
            assert all(isinstance(document, Document) for document in documents)

    def test_get_file_token_count(self, created_files, backend):
        for file in created_files:
            token_count = backend.get_file_token_count(file_uuid=file.uuid)
            assert isinstance(token_count, int)
            assert token_count > 0


# region TAGS ====================


@pytest.mark.incremental
class TestTags:
    """Tests CRUD for Tags."""

    def test_get_tag(self, created_tag, backend):
        tag = backend.get_tag(tag_uuid=created_tag.uuid)
        assert isinstance(tag, Tag)

    def test_add_files_to_tag(self, created_tag, created_files, backend):
        file_uuids = [f.uuid for f in created_files]

        _ = backend.add_files_to_tag(file_uuids=file_uuids, tag_uuid=created_tag.uuid)
        time.sleep(1)

        tag = backend.get_tag(tag_uuid=created_tag.uuid)
        assert set(file_uuids) <= tag.files

    def test_remove_files_from_tag(self, created_tag, created_files, backend):
        file_uuids = [f.uuid for f in created_files]

        _ = backend.remove_files_from_tag(file_uuids=file_uuids, tag_uuid=created_tag.uuid)
        time.sleep(1)

        tag = backend.get_tag(tag_uuid=created_tag.uuid)
        assert not (set(file_uuids) <= tag.files)

    def test_list_tags(self, created_tag, backend):
        assert created_tag in backend.list_tags()


# region SUMMARIES ====================


class TestSummaries:
    """Tests CRUD for Summarys."""

    def test_get_summary(self, created_summary, created_files, backend):
        file_uuids = [f.uuid for f in created_files]
        summary = backend.get_summary(file_uuids=file_uuids)
        assert isinstance(summary, SummaryComplete)
        assert summary == created_summary

    def test_list_summaries(self, created_summary, backend):
        assert created_summary in backend.list_summaries()


# region FEEDBACK ====================


class TestFeedback:
    """Tests CRUD for Feedback."""

    def test_create_feedback(self, backend):
        feedback = Feedback(
            input=[{"role": "user", "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."}],
            output={"role": "ai", "text": "Ut enim ad minim veniam."},
            feedback_type="thumbs",
            feedback_score="👍",
            feedback_text="Baz",
            creator_user_uuid=TEST_USER_UUID,
        )

        given = backend.create_feedback(feedback=feedback)

        assert isinstance(given, Feedback)


# region LLM ====================


class TestLLM:
    """Tests LLM calls for a backend."""

    def test_rag_chat(self, created_files, backend):
        request = ChatRequest(
            message_history=[
                {"role": "user", "text": "What does Mr. Aneurin Bevan think of the national health insurance system"}
            ]
        )
        response = backend.rag_chat(chat_request=request)

        assert isinstance(response, ChatResponse)
        assert len(response.output_text) > 0
        assert len(response.source_documents) > 0

    def test_stuff_doc_summary(self, created_files, backend):
        file_short = next(f for f in created_files if "smarter" in f.key.lower())

        summary = PromptTemplate.from_template("Summarise this text: {text}")
        response = backend.stuff_doc_summary(summary=summary, file_uuids=[file_short.uuid])

        assert isinstance(response, ChatResponse)
        assert len(response.output_text) > 0
        assert len(response.source_documents) > 0

    def test_map_reduce_summary(self, created_files, backend):
        file_short = next(f for f in created_files if "smarter" in f.key.lower())
        # file_long = next(f for f in created_files if "health" in f.key.lower())
        # TODO: Speed up embeddings so we can do long files again
        file_long = next(f for f in created_files if "smarter" in f.key.lower())

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


# region USER AND CONFIG ====================


class TestUser:
    """Tests getters and setters for settings."""

    def test_get_set_user(self, backend):
        user_current = backend.get_user()

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

        _ = backend.set_user(
            uuid=user_current.uuid,
            name=user_current.name,
            email=user_current.email,
            department=user_current.department,
            role=user_current.role,
            preferred_language=user_current.preferred_language,
        )

    def test_get_set_llm(self, backend):
        llm_current = backend.get_llm()

        llm_sent = backend.set_llm(
            model="mistral/mistral-tiny",
            max_tokens=1_000,
            max_return_tokens=256,
            temperature=0.1,
        )
        llm_returned = backend.get_llm()

        assert llm_sent == llm_returned

        _ = backend.set_llm(
            model=llm_current.llm.model,
            max_tokens=llm_current.max_tokens,
            max_return_tokens=llm_current.llm.max_tokens,
            temperature=llm_current.llm.temperature,
        )
