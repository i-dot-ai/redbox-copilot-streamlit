import re
import unicodedata
from typing import Optional
from uuid import UUID

import streamlit as st

from redbox.models import ChatMessage, ChatMessageSourced, ChatResponse, ChatSource, File, FileStatus, SourceDocument


def get_link_html(page: str, text: str, query_dict: Optional[dict] = None, target: str = "_self") -> str:
    """Returns a link in HTML format

    Args:
        page (str): the page to link to
        text (str): the text to display
        query_dict (dict, optional): query parameters. Defaults to None.
        target (str, optional): the target of the link. Defaults to "_self".

    Returns:
        str: _description_
    """
    if query_dict is not None:
        query = "&".join(f"{k}={v}" for k, v in query_dict.items())
        query = "?" + query
    else:
        query = ""

    return (
        f"<a href='/{page}{query}' target={target}>"
        f"<button style='background-color: white;border-radius: 8px;'>{text}</button>"
        "</a>"
    )


def get_file_link(file: File, page: Optional[int] = None) -> str:
    """Returns a link to a file

    Args:
        file (File): the file to link to
        page (int, optional): the page to link to in the file. Defaults to None.

    Returns:
        _type_: _description_
    """
    # we need to refer to files by their uuid instead
    if len(file.key) > 45:
        presentation_name = file.key[:45] + "..."
    else:
        presentation_name = file.key

    query_dict = {"file_uuid": str(file.uuid)}
    if page is not None:
        query_dict["page_number"] = str(page)

    link_html = get_link_html(
        page="Documents",
        query_dict=query_dict,
        text=presentation_name,
    )

    return link_html


def get_document_citation_assets(document: SourceDocument) -> tuple[File, Optional[int | list[int]], str]:
    """Takes a SourceDocument and returns a tuple of its File, page numbers and URL."""
    file = st.session_state.backend.get_file(file_uuid=document.file_uuid)

    if isinstance(document.page_numbers, int):
        url = get_file_link(file=file, page=document.page_numbers)
    elif isinstance(document.page_numbers, list):
        url = get_file_link(file=file, page=document.page_numbers[0])
    else:
        url = get_file_link(file=file)

    return file, document.page_numbers, url


def response_to_message(response: ChatResponse) -> ChatMessage | ChatMessageSourced:
    sources: list[ChatSource] = []
    if response.source_documents is not None:
        for source in response.source_documents:
            if not isinstance(source, SourceDocument):
                raise ValueError("Got SourceDocument of type %s", type(source))

            chat_source = ChatSource(document=source, html=get_document_citation_assets(source)[2])
            sources.append(chat_source)
        return ChatMessageSourced(text=response.output_text, role="ai", sources=sources)
    else:
        return ChatMessage(text=response.output_text, role="ai")


def format_feedback_kwargs(chat_history: list[ChatMessage], user_uuid: UUID, n: int = -1) -> dict:
    """Formats feedback kwarg dict based on a chat history."""
    return {
        "input": chat_history[0:n],
        "output": chat_history[n],
        "creator_user_uuid": user_uuid,
    }


def slugify(text: str) -> str:
    slug = unicodedata.normalize("NFKD", text)
    slug = slug.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    slug = re.sub(r"[-]+", "-", slug)
    return slug


def format_file_status(status: FileStatus) -> str:
    status_title = status.processing_status.title()
    n_chunks = 0
    if status.chunk_statuses is not None:
        n_chunks = len(status.chunk_statuses)

    if status.processing_status == "embedding" and n_chunks > 0 and status.chunk_statuses is not None:
        n_chunks_embedded = len([chunk for chunk in status.chunk_statuses if chunk.embedded])
        return f"{status_title} ({n_chunks_embedded / n_chunks:.0%})"
    else:
        return status_title
