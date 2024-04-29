from io import BytesIO
from uuid import UUID
from pathlib import Path

import pytest

from redbox.models import UploadFile, ContentType, File
from redbox.tests.conftest import TEST_DATA, YieldFixture
from redbox.local import LocalBackendAdapter
from redbox.definitions import BackendAdapter

DOCS = [Path(*x.parts[-2:]) for x in (TEST_DATA / "docs").glob("*.*")]
ADAPTERS = [LocalBackendAdapter]


@pytest.mark.incremental
@pytest.mark.parametrize("doc", DOCS, scope="class")
@pytest.mark.parametrize("adapter", ADAPTERS, scope="class")
class TestFiles:
    """Tests CRUD for a backend."""

    @pytest.fixture(scope="class")
    def backend(self, settings, adapter) -> YieldFixture[BackendAdapter]:
        backend = adapter(settings=settings)
        backend._set_uuid(user_uuid=UUID("bd65600d-8669-4903-8a14-af88203add38"))
        backend._set_llm(
            model="openai/gpt-3.5-turbo",
            max_tokens=1024,
            temperature=0.2,
        )
        yield backend

    @pytest.fixture(scope="class")
    def file(self, doc) -> YieldFixture[UploadFile]:
        full_path = Path(TEST_DATA / doc)
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

    def test_create_file(self, backend, file):
        file = backend.create_file(file=file)
        assert isinstance(file, File)
