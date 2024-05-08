from typing import TYPE_CHECKING

from langchain_core.embeddings.embeddings import Embeddings

from redbox.models.file import Chunk, ContentType, File
from redbox.parsing.chunk_clustering import cluster_chunks
from redbox.parsing.chunkers import other_chunker

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object


class FileChunker:
    """A class to wrap unstructured and generate compliant chunks from files"""

    def __init__(self, s3_client: S3Client, bucket_name: str, embedding_model: Embeddings):
        self.supported_file_types = [content_type.value for content_type in ContentType]
        self.embedding_model = embedding_model
        self.s3_client = s3_client
        self.bucket_name = bucket_name

    def chunk_file(
        self,
        file: File,
        chunk_clustering: bool = True,
    ) -> list[Chunk]:
        """_summary_

        Args:
            file (File): The file to read, analyse layout and chunk.
            file_url (str): The authenticated url of the file to fetch, analyse layout and chunk.
            chunk_clustering (bool): Whether to merge small semantically similar chunks.
                Defaults to True.
        Raises:
            ValueError: Will raise when a file is not supported.

        Returns:
            List[Chunk]: The chunks generated from the given file.
        """
        chunks = other_chunker(file=file, s3_client=self.s3_client, bucket_name=self.bucket_name)

        if chunk_clustering:
            chunks = cluster_chunks(chunks, embedding_model=self.embedding_model)

        return chunks
