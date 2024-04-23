from langchain.prompts.prompt import PromptTemplate
import logging
from redbox.llm.prompts.core import _core_redbox_prompt
from redbox.models.chat import ChatMessage

_chat_template = """Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question, in its original
language. include the follow up instructions in the standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_chat_template)


_with_sources_template = """Given the following extracted parts of a long document and \
a question, create a final answer with Sources at the end.  \
If you don't know the answer, just say that you don't know. Don't try to make \
up an answer.
Be concise in your response and summarise where appropriate. \
At the end of your response add a "Sources:" section with the documents you used. \
DO NOT reference the source documents in your response. Only cite at the end. \
ONLY PUT CITED DOCUMENTS IN THE "Sources:" SECTION AND NO WHERE ELSE IN YOUR RESPONSE. \
IT IS CRUCIAL that citations only happens in the "Sources:" section. \
This format should be <DocX> where X is the document UUID being cited.  \
DO NOT INCLUDE ANY DOCUMENTS IN THE "Sources:" THAT YOU DID NOT USE IN YOUR RESPONSE. \
YOU MUST CITE USING THE <DocX> FORMAT. NO OTHER FORMAT WILL BE ACCEPTED.
Example: "Sources: <DocX> <DocY> <DocZ>"

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

from typing import Callable, Optional
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatLiteLLM

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def get_chat_runnable(
    llm: ChatLiteLLM,
    get_history_func: Callable,
    init_messages: Optional[list[ChatMessage]] = None,
) -> RunnableWithMessageHistory:
    if init_messages is None:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
    else:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                *((msg["role"], msg["text"]) for msg in init_messages),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

    runnable = chat_prompt | llm

    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_history_func,
        input_messages_key="input",
        history_messages_key="history",
    )

    return with_message_history
