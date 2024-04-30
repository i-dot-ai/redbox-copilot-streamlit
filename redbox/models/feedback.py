from typing import Optional

from langchain.chains.base import Chain
from langchain_core.documents.base import Document
from pydantic import field_serializer, field_validator

from redbox.models.base import PersistableModel


class Feedback(PersistableModel):
    input: str | list[str]
    # langchain.chains.base.Chain needs pydantic v1, breaks
    # https://python.langchain.com/docs/guides/pydantic_compatibility
    chain: Optional[object] = None
    output: str
    # langchain_core.documents.base.Document needs pydantic v1, breaks
    # https://python.langchain.com/docs/guides/pydantic_compatibility
    sources: Optional[list[object]] = None
    feedback_type: str
    feedback_score: str
    feedback_text: Optional[str] = None

    @field_validator("sources")
    @classmethod
    def is_document(cls, v: list[Document]) -> list[Document]:
        """Custom validator for using Pydantic v1 object in model."""
        doc_list: list[Document] = []
        for doc in v:
            if not isinstance(doc, Document):
                doc_list.append(Document(**doc))
            else:
                doc_list.append(v)

        assert all(isinstance(doc, Document) for doc in doc_list)
        return doc_list

    @field_serializer("chain")
    def serialise_chain(self, chain: Chain, _info):
        if isinstance(chain, Chain):
            return chain.dict()
        else:
            return chain

    @field_serializer("sources")
    def serialise_doc(self, sources: list[Document], _info):
        return [source.dict() for source in sources]
