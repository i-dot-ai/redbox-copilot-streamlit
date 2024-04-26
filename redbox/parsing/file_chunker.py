from langchain_core.embeddings.embeddings import Embeddings

from redbox.models.file import Chunk, ContentType, File
from redbox.parsing.chunk_clustering import cluster_chunks
from redbox.parsing.chunkers import other_chunker
from botocore.client import BaseClient


class FileChunker:
    """A class to wrap unstructured and generate compliant chunks from files"""

    def __init__(self, s3_client: BaseClient, bucket_name: str, embedding_model: Embeddings = None):
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

        # Ensure page numbers are a list for schema compliance
        for chunk in chunks:
            if "page_number" in chunk.metadata:
                if isinstance(chunk.metadata["page_number"], int):
                    chunk.metadata["page_numbers"] = [chunk.metadata["page_number"]]
                elif isinstance(chunk.metadata["page_number"], list):
                    chunk.metadata["page_numbers"] = chunk.metadata["page_number"]
                del chunk.metadata["page_number"]

        return chunks
