import logging
from uuid import uuid4

from sentence_transformers import SentenceTransformer

from redbox.models import ModelInfo
from redbox.models.llm import Embedding, EmbeddingResponse

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class SentenceTransformerDB(SentenceTransformer):
    def __init__(self, model_name: str, cache_folder: str):
        super().__init__(model_name_or_path=model_name, cache_folder=cache_folder)
        self.model_name = model_name

    def embed_sentences(self, sentences: list[str]) -> EmbeddingResponse:
        embeddings = self.encode(sentences)

        reformatted_embeddings = [
            Embedding(
                object="embedding",
                index=i,
                embedding=list(embedding),
            )
            for i, embedding in enumerate(embeddings)
        ]

        output = EmbeddingResponse(
            object="list",
            data=reformatted_embeddings,
            embedding_id=str(uuid4()),
            embedding_model=self.model_name,
            embedding_model_info=self.get_model_info(),
        )

        return output

    def get_model_info(self) -> ModelInfo:
        model_info = ModelInfo(
            model=self.model_name,
            max_seq_length=self.get_max_seq_length(),
            vector_size=self.get_sentence_embedding_dimension(),
        )
        return model_info
