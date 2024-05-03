import base64
import os
from uuid import UUID
from datetime import datetime
from io import BytesIO
from typing import Optional
from uuid import UUID
import unicodedata
import re

import dotenv
import html2markdown
import pandas as pd
import streamlit as st
from langchain.callbacks import FileCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import LLMResult
from loguru import logger
from lxml.html.clean import Cleaner

from redbox.definitions import BackendAdapter
from redbox.local import LocalBackendAdapter
from redbox.models import (
    ChatMessage,
    ChatMessageSourced,
    ChatResponse,
    ChatSource,
    Feedback,
    File,
    SourceDocument,
    Settings
)
from redbox.local import LocalBackendAdapter
from redbox.definitions import BackendAdapter

DEV_UUID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


def init_session_state() -> dict:
    """
    Initialise the session state and return the environment variables

    Returns:
        dict: the environment variables dictionary
    """
    # Bring VARS into environment from any .env file
    DOT_ENV = dotenv.dotenv_values(".env")
    # Grab it as a dictionary too for convenience
    ENV = dict(os.environ)
    # Update the environment with the .env file
    if DOT_ENV:
        ENV.update(DOT_ENV)  # type: ignore[arg-type]

    st.markdown(
        """
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
    </style>
    """,
        unsafe_allow_html=True,
    )

    if "available_models" not in st.session_state:
        st.session_state.available_models = []

        if "OPENAI_API_KEY" in ENV:
            if ENV["OPENAI_API_KEY"]:
                st.session_state.available_models.append("openai/gpt-3.5-turbo")

        if "ANTHROPIC_API_KEY" in ENV:
            if ENV["ANTHROPIC_API_KEY"]:
                st.session_state.available_models.append("anthropic/claude-2")

        if len(st.session_state.available_models) == 0:
            st.error("No models available. Please check your API keys.")
            st.stop()

    if "model_select" not in st.session_state:
        st.session_state.model_select = st.session_state.available_models[0]

    if ENV["DEV_MODE"]:
        st.sidebar.info("**DEV MODE**")
        with st.sidebar.expander("⚙️ DEV Settings", expanded=False):
            st.session_state.model_params = {
                # TODO: This shoudld be dynamic to the model
                "max_tokens": st.number_input(
                    label="max_tokens",
                    min_value=0,
                    max_value=100_000,
                    value=10_000,
                    step=1,
                ),
                "max_return_tokens": st.number_input(
                    label="max_tokens",
                    min_value=0,
                    max_value=5_000,
                    value=1024,
                    step=1,
                ),
                "temperature": st.slider(
                    label="temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.01,
                ),
            }
            reload_llm = st.button(label="♻️ Reload LLM and LLMHandler")
            if reload_llm:
                st.session_state.backend.set_llm(
                    model=st.session_state.model_select,
                    max_tokens=st.session_state.model_params["max_tokens"],
                    max_return_tokens=st.session_state.model_params["max_return_tokens"],
                    temperature=st.session_state.model_params["temperature"],
                )

            if st.button(label="Empty Streamlit Cache"):
                st.cache_data.clear()
    else:
        _model_params = {"max_tokens": 10_000, "max_return_tokens": 1_000, "temperature": 0.2}

    if "backend" not in st.session_state:
        st.session_state.backend = LocalBackendAdapter(settings=Settings())

        st.session_state.backend.set_user(
            uuid=DEV_UUID,
            name="",
            email="",
            department="Cabinet Office",
            role="Civil Servant",
            preferred_language="British English",
        )

        st.session_state.backend.set_llm(
            model=st.session_state.model_select,
            max_tokens=st.session_state.model_params["max_tokens"],
            max_return_tokens=st.session_state.model_params["max_return_tokens"],
            temperature=st.session_state.model_params["temperature"],
        )

    if "llm_logger_callback" not in st.session_state:
        logfile = os.path.join(
            "llm_logs",
            f"llm_{datetime.now().isoformat(timespec='seconds')}.log",
        )
        logger.add(logfile, colorize=True, enqueue=True)
        st.session_state.llm_logger_callback = FileCallbackHandler(logfile)

    return ENV


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
    if len(file.name) > 45:
        presentation_name = file.name[:45] + "..."
    else:
        presentation_name = file.name

    query_dict = {"file_uuid": str(file.uuid)}
    if page is not None:
        query_dict["page_number"] = str(page)

    link_html = get_link_html(
        page="Preview_Files",
        query_dict=query_dict,
        text=presentation_name,
    )

    return link_html


class StreamlitStreamHandler(BaseCallbackHandler):
    """Callback handler for rendering LLM output to streamlit UI"""

    def __init__(self, text_element, initial_text=""):
        self.text_element = text_element
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Callback for new token from LLM and append to text"""
        del kwargs
        self.text += token
        self.text_element.write(self.text)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Callback for end of LLM generation to empty text"""
        del kwargs
        self.text_element.empty()

    def sync(self):
        """Syncs the text element with the text"""
        self.text_element.write(self.text)


class FilePreview(object):
    """Class for rendering files to streamlit UI"""

    def __init__(self, backend: BackendAdapter):
        self.cleaner = Cleaner()
        self.cleaner.javascript = True
        self.backend = backend

        self.render_methods = {
            ".pdf": self._render_pdf,
            ".txt": self._render_txt,
            ".xlsx": self._render_xlsx,
            ".csv": self._render_csv,
            ".eml": self._render_eml,
            ".html": self._render_html,
            ".docx": self._render_docx,
        }

    def st_render(self, file: File) -> None:
        """Outputs the given file to streamlit UI

        Args:
            file (File): The file to preview
        """

        render_method = self.render_methods[file.content_type]
        file_bytes = self.backend.get_object(file_uuid=file.uuid)
        render_method(file, file_bytes)

    def _render_pdf(self, file: File, page_number: Optional[int] = None) -> None:
        file_bytes = self.backend.get_object(file_uuid=file.uuid)
        base64_pdf = base64.b64encode(file_bytes).decode("utf-8")

        if page_number is not None:
            iframe = f"""<iframe
                        title="{file.name}" \
                        src="data:application/pdf;base64,{base64_pdf}#page={page_number}" \
                        width="100%" \
                        height="1000" \
                        type="application/pdf"></iframe>"""
        else:
            iframe = f"""<iframe
                        title="{file.name}" \
                        src="data:application/pdf;base64,{base64_pdf}" \
                        width="100%" \
                        height="1000" \
                        type="application/pdf"></iframe>"""

        st.markdown(iframe, unsafe_allow_html=True)

    def _render_txt(self, file: File, file_bytes: bytes) -> None:
        st.markdown(f"{file_bytes.decode('utf-8')}", unsafe_allow_html=True)

    def _render_xlsx(self, file: File, file_bytes: bytes) -> None:
        df = pd.read_excel(BytesIO(file_bytes))
        st.dataframe(df, use_container_width=True)

    def _render_csv(self, file: File, file_bytes: bytes) -> None:
        df = pd.read_csv(BytesIO(file_bytes))
        st.dataframe(df, use_container_width=True)

    def _render_eml(self, file: File, file_bytes: bytes) -> None:
        st.markdown(self.cleaner.clean_html(file_bytes.decode("utf-8")), unsafe_allow_html=True)

    def _render_html(self, file: File, file_bytes: bytes) -> None:
        markdown_html = html2markdown.convert(file_bytes.decode("utf-8"))
        st.markdown(markdown_html, unsafe_allow_html=True)

    def _render_docx(self, file: File, file_bytes: bytes) -> None:
        st.warning("DOCX preview not yet supported.")
        st.download_button(
            label=file.name,
            data=file_bytes,
            mime="application/msword",
            file_name=file.name,
        )


def submit_feedback(
    feedback: dict[str, str],
    input: ChatMessage | list[ChatMessage],
    output: ChatMessage,
    creator_user_uuid: UUID,
) -> None:
    """Submits feedback to the storage handler
    Args:
        feedback (Dict): A dictionary containing the feedback
        input (Union[str, List[str]]): Input text from the user
        output (str): The output text from the LLM
        sources (list): The sources that were used
        creator_user_uuid (str): The uuid of the user who created the feedback
        chain (Optional[Chain], optional): The chain used to generate the output. Defaults to None.
    """
    feedback = Feedback(
        input=input,
        output=output,
        feedback_type=feedback["type"],
        feedback_score=feedback["score"],
        feedback_text=feedback["text"],
        creator_user_uuid=creator_user_uuid,
    )

    st.session_state.backend.create_feedback(feedback=feedback)

    st.toast("Thanks for your feedback!", icon="🙏")


def get_document_citation_assets(document: SourceDocument) -> set[tuple[File, Optional[list[int]], str]]:
    """Takes a SourceDocument and returns a tuple of its File, page numbers and URL."""
    file = st.session_state.backend.get_file(file_uuid=document.file_uuid)

    if document.page_numbers is None:
        url = get_file_link(file=file)
    else:
        for page in document.page_numbers:
            url = get_file_link(file=file, page=page)

    return file, document.page_numbers, url


def response_to_message(response: ChatResponse) -> ChatMessage | ChatMessageSourced:
    sources: list[ChatSource] = []
    if response.source_documents is not None:
        for source in response.source_documents:
            sources.append(ChatSource(document=source, html=get_document_citation_assets(source)[2]))
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


def change_selected_model() -> None:
    st.session_state.backend.set_llm(
        model=st.session_state.model_select,
        max_tokens=st.session_state.model_params["max_tokens"],
        max_return_tokens=st.session_state.model_params["max_return_tokens"],
        temperature=st.session_state.model_params["temperature"],
    )
    st.toast(f"Loaded {st.session_state.model_select}")


def slugify(text: str) -> str:
    slug = unicodedata.normalize("NFKD", text)
    slug = slug.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    slug = re.sub(r"[-]+", "-", slug)
    return slug
