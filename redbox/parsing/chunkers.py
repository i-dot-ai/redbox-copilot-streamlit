import logging
from typing import TYPE_CHECKING

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from redbox.models import Chunk, File, Metadata

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

logging.basicConfig(level=logging.INFO)


def other_chunker(file: File, s3_client: S3Client, bucket_name: str) -> list[Chunk]:
    authenticated_s3_url = s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": file.key},
        ExpiresIn=3600,
    )

    elements = partition(url=authenticated_s3_url)
    raw_chunks = chunk_by_title(elements=elements)

    chunks = []
    for i, raw_chunk in enumerate(raw_chunks):
        raw_chunk = raw_chunk.to_dict()
        raw_chunk["metadata"]["parent_doc_uuid"] = file.uuid

        page_number = raw_chunk["metadata"].get("page_number")
        if isinstance(page_number, int):
            raw_chunk["metadata"]["page_number"] = [page_number]

        metadata = Metadata(**raw_chunk["metadata"])

        chunk = Chunk(
            parent_file_uuid=file.uuid,
            index=i,
            text=raw_chunk["text"],
            metadata=metadata,
            creator_user_uuid=file.creator_user_uuid,
        )
        chunks.append(chunk)

    return chunks
