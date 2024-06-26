from datetime import date
from typing import Any, Optional

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from redbox.llm.prompts.chat import (
    CONDENSE_QUESTION_PROMPT,
    STUFF_DOCUMENT_PROMPT,
    WITH_SOURCES_PROMPT,
)


class LLMHandler(object):
    """A class to handle RedBox data suffused interactions with a given LLM"""

    def __init__(
        self,
        llm,
        user_uuid: str,
        max_tokens: int,
        vector_store=None,
        embedding_function: Optional[HuggingFaceEmbeddings] = None,
    ):
        """Initialise LLMHandler

        Args:
            llm (_type_): _description_
            user_uuid: Session to load data from and save data to.
            max_tokens: The max size of this LLM's context window
            vector_store (Optional[Chroma], optional): _description_.
            Defaults to None.
            embedding_function (Optional[HuggingFaceEmbeddings], optional):
            _description_. Defaults to None.
        """
        self.llm = llm
        self.user_uuid = user_uuid
        self.max_tokens = max_tokens
        self.embedding_function = embedding_function
        self.vector_store = vector_store

    def chat_with_rag(
        self,
        user_question: str,
        user_info: dict,
        chat_history: Optional[str] = None,
        callbacks: Optional[list] = None,
    ) -> dict[str, Any]:
        """Answers user question by retrieving context from content stored in
        Vector DB

        Args:
            user_question (str): The message or query being posed by user
            chat_history (str, optional): The message history of the chat to
            add context. Defaults to an empty string.

        Returns:
            dict: A dictionary with the new chat_history:list and the answer
            BaseCombineDocumentsChain: docs-with-sources-chain
        """

        docs_with_sources_chain = load_qa_with_sources_chain(
            self.llm,
            chain_type="stuff",
            prompt=WITH_SOURCES_PROMPT,
            document_prompt=STUFF_DOCUMENT_PROMPT,
            verbose=True,
        )

        condense_question_chain = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)

        # split chain manually, so that the standalone question doesn't leak into chat
        # should we display some waiting message instead?
        standalone_question = condense_question_chain(
            {
                "question": user_question,
                "chat_history": chat_history or "",
                # "user_info": user_info,
                # "current_date": date.today().isoformat()
            }
        )["text"]

        docs = self.vector_store.as_retriever().get_relevant_documents(
            standalone_question,
        )

        result = docs_with_sources_chain(
            {
                "question": standalone_question,
                "input_documents": docs,
                "user_info": user_info,
                "current_date": date.today().isoformat(),
            },
            callbacks=callbacks or [],
        )
        return result

    def stuff_doc_summary(
        self, prompt: PromptTemplate, documents: list[Document], user_info: dict, callbacks: Optional[list] = None
    ):
        summary_chain = LLMChain(llm=self.llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=summary_chain, document_variable_name="text")

        result = stuff_chain.run(
            user_info=user_info,
            current_date=date.today().isoformat(),
            input_documents=documents,
            callbacks=callbacks or [],
        )

        return result

    def map_reduce_summary(
        self,
        map_prompt: PromptTemplate,
        reduce_prompt: PromptTemplate,
        documents: list[Document],
        user_info: dict,
        callbacks: Optional[list] = None,
    ):
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="text")
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=self.max_tokens,
        )
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="text",
            return_intermediate_steps=False,
        )

        result = map_reduce_chain.run(
            user_info=user_info,
            current_date=date.today().isoformat(),
            input_documents=documents,
            callbacks=callbacks or [],
        )

        return result
