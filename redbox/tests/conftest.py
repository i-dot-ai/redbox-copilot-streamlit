from typing import Generator, TypeVar, Dict, Tuple
from uuid import uuid4
from pathlib import Path

import pytest
from elasticsearch import Elasticsearch


from redbox.models import Chunk, File, Settings
from redbox.storage.elasticsearch import ElasticsearchStorageHandler

ROOT = Path(__file__).parent.parent.parent
TEST_DATA = Path(ROOT / "redbox/tests/data")

T = TypeVar("T")
YieldFixture = Generator[T, None, None]

_test_failed_incremental: Dict[str, Dict[Tuple[int, ...], str]] = {}
"""Incremental hook pattern fixture.

https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps
"""


def pytest_runtest_makereport(item, call):
    """Incremental hook pattern fixture.

    https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps
    """
    if "incremental" in item.keywords:
        if call.excinfo is not None:
            cls_name = str(item.cls)
            parametrize_index = tuple(item.callspec.indices.values()) if hasattr(item, "callspec") else ()
            test_name = item.originalname or item.name
            _test_failed_incremental.setdefault(cls_name, {}).setdefault(parametrize_index, test_name)


def pytest_runtest_setup(item):
    """Incremental hook pattern fixture.

    https://docs.pytest.org/en/7.1.x/example/simple.html#incremental-testing-test-steps
    """
    if "incremental" in item.keywords:
        cls_name = str(item.cls)
        if cls_name in _test_failed_incremental:
            parametrize_index = tuple(item.callspec.indices.values()) if hasattr(item, "callspec") else ()
            test_name = _test_failed_incremental[cls_name].get(parametrize_index, None)
            if test_name is not None:
                pytest.xfail("previous test failed ({})".format(test_name))


@pytest.fixture(scope="session")
def settings() -> YieldFixture[Settings]:
    settings = Settings(_env_file=Path(ROOT, ".env.test"), _env_file_encoding="utf-8")
    assert Path(ROOT, ".env.test").exists()
    assert settings.minio_host == "localhost"
    yield settings


@pytest.fixture
def chunk() -> Chunk:
    test_chunk = Chunk(
        parent_file_uuid=uuid4(),
        index=1,
        text="test_text",
        metadata={},
    )
    return test_chunk


@pytest.fixture
def another_chunk() -> Chunk:
    test_chunk = Chunk(
        parent_file_uuid=uuid4(),
        index=1,
        text="test_text",
        metadata={},
    )
    return test_chunk


@pytest.fixture
def file() -> File:
    test_file = File(
        name="test.pdf",
        url="http://example.com/test.pdf",
        status="uploaded",
        content_type=".pdf",
    )
    return test_file


@pytest.fixture
def stored_chunk(elasticsearch_storage_handler, chunk) -> Chunk:
    elasticsearch_storage_handler.write_item(item=chunk)
    return chunk


@pytest.fixture
def elasticsearch_client(settings) -> YieldFixture[Elasticsearch]:
    yield settings.elasticsearch_client()


@pytest.fixture
def elasticsearch_storage_handler(elasticsearch_client):
    yield ElasticsearchStorageHandler(es_client=elasticsearch_client, root_index="redbox-test-data")
