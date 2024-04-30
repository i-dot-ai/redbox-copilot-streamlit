from typing import Optional

from langchain.chains.base import Chain
from langchain_core.documents.base import Document
from pydantic import field_serializer

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

    @field_serializer("chain")
    def serialise_chain(self, chain: Chain, _info):
        if isinstance(chain, Chain):
            return chain.dict()
        else:
            return chain

    @field_serializer("sources")
    def serialise_doc(self, sources: list[Document], _info):
        serialisable: list[dict] = []

        if isinstance(sources, Document):
            serialisable.append(sources.dict())
        elif isinstance(sources, list):
            for doc in sources:
                if isinstance(doc, Document):
                    serialisable.append(doc.dict())
                else:
                    serialisable.append(doc)
        else:
            serialisable = sources

        return serialisable
