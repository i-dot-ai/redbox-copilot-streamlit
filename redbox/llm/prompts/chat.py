from operator import itemgetter
from langchain.prompts.prompt import PromptTemplate
import logging
from redbox.llm.prompts.core import _core_redbox_prompt
from redbox.models.chat import ChatMessage

from typing import Callable, Optional
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

_chat_template = """Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question, in its original
language. include the follow up instructions in the standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_chat_template)

_with_sources_template = """Given the following extracted parts of a long document and \
a question, create a final answer.  \
If you don't know the answer, just say that you don't know. Don't try to make \
up an answer.
Be concise in your response and summarise where appropriate. \

Use **bold** to highlight the most question relevant parts in your response.
If dealing dealing with lots of data return it in markdown table format.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

WITH_SOURCES_PROMPT = PromptTemplate.from_template(_core_redbox_prompt + _with_sources_template)

_stuff_document_template = "<Doc{parent_doc_uuid}>{page_content}</Doc{parent_doc_uuid}>"

STUFF_DOCUMENT_PROMPT = PromptTemplate.from_template(_stuff_document_template)


def get_chat_runnable(
    llm: ChatLiteLLM,
    get_history_func: Callable,
    init_messages: Optional[list[ChatMessage]] = None,
) -> RunnableWithMessageHistory:
    if init_messages is None:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("placeholder", "{history}"),
                ("human", "{input}"),
            ]
        )
    else:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                *((msg["role"], msg["text"]) for msg in init_messages),
                ("placeholder", "{history}"),
                ("human", "{input}"),
            ]
        )

    runnable = chat_prompt | llm | StrOutputParser()

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history=get_history_func,
        input_messages_key="input",
        history_messages_key="history",
    )

    return with_message_history


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_runnable(
    llm: ChatLiteLLM, get_history_func: Callable, retriever: VectorStoreRetriever
) -> RunnableWithMessageHistory:
    prompt = ChatPromptTemplate.from_messages(
        [("system", _core_redbox_prompt), ("placeholder", "{history}"), ("human", _with_sources_template)]
    )
    context = itemgetter("question") | retriever
    setup = RunnablePassthrough.assign(summaries=context | format_docs, sources=context)
    runnable = setup | {"response": prompt | llm, "sources": itemgetter("sources")}

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history=get_history_func,
        input_messages_key="question",
        output_messages_key="response",
        history_messages_key="history",
    )

    return with_message_history
