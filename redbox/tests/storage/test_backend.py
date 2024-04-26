from io import BytesIO
from uuid import UUID
from pathlib import Path

import pytest

from redbox.models import UploadFile, ContentType, File
from redbox.tests.conftest import TEST_DATA, YieldFixture

DOCS = [Path(*x.parts[-2:]) for x in (TEST_DATA / 'docs').glob('*.html')]

@pytest.mark.incremental
@pytest.mark.parametrize("doc", DOCS)
class TestFiles():
    """Tests CRUD for a backend."""

    @pytest.fixture(scope="class")
    def file(doc) -> YieldFixture[UploadFile]:
        full_path = Path(TEST_DATA / doc)
        sanitised_name = full_path.name.replace("'", "_")
        file_type = full_path.suffix

        with open(full_path, 'rb') as f:
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
