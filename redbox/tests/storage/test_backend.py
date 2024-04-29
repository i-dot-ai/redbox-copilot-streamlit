from io import BytesIO
from uuid import UUID
from pathlib import Path
from typing import Optional
import time

import pytest

from elasticsearch import NotFoundError

from redbox.models import UploadFile, ContentType, File
from redbox.tests.conftest import TEST_DATA, YieldFixture
from redbox.local import LocalBackendAdapter
from redbox.definitions import BackendAdapter

DOCS = [Path(*x.parts[-2:]) for x in (TEST_DATA / "docs").glob("*.txt")]
ADAPTERS = [LocalBackendAdapter]


@pytest.mark.incremental
class TestFiles:
    """Tests CRUD for a backend."""

    file: Optional[File] = None
    files: Optional[list[File]] = []

    @pytest.fixture(params=ADAPTERS)
    def backend(self, request, settings) -> YieldFixture[BackendAdapter]:
        backend = request.param(settings=settings)
        backend._set_uuid(user_uuid=UUID("bd65600d-8669-4903-8a14-af88203add38"))
        backend._set_llm(
            model="openai/gpt-3.5-turbo",
            max_tokens=1024,
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

    def test_delete_file(self, backend):
        file = backend.delete_file(file_uuid=self.file.uuid)
        time.sleep(3)
        assert file.uuid not in [f.uuid for f in backend.list_files()]

        chunks = backend.get_file_chunks(file_uuid=file.uuid)
        assert len(chunks) == 0

        with pytest.raises(NotFoundError):
            _ = backend.get_object(file_uuid=file.uuid)
