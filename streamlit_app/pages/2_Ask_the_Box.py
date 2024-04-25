import json
from uuid import UUID, uuid4
from datetime import date, datetime
import itertools


import streamlit as st
from streamlit_feedback import streamlit_feedback

from redbox.llm.prompts.core import CORE_REDBOX_PROMPT
from redbox.models.chat import ChatMessage
from streamlit_app.utils import (
    init_session_state,
    submit_feedback,
    get_stream_key,
    render_document_citations,
    change_selected_model,
)

st.set_page_config(page_title="Redbox Copilot - Ask the Box", page_icon="📮", layout="wide")

# region Global and session state variables, functions ====================

with st.spinner("Loading..."):
    ENV = init_session_state()
    USER = st.session_state.backend.get_user()
    INITIAL_CHAT_PROMPT = [
        ChatMessage(
            text=CORE_REDBOX_PROMPT.format(
                current_date=date.today().isoformat(),
                user_info=USER.summary(),
            ),
            role="system",
        ),
        ChatMessage(text="Hi, I'm Redbox Copilot. How can I help you?", role="ai"),
    ]
    AVATAR_MAP = {
        "human": "🧑‍💻",
        "ai": "📮",
        "user": "🧑‍💻",
        "assistant": "📮",
        "AIMessageChunk": "📮",
    }
    FEEDBACK_KWARGS = {
        "feedback_type": "thumbs",
        "optional_text_label": "What did you think of this response?",
        "on_submit": submit_feedback,
    }


def clear_chat() -> None:
    st.session_state.chat_id = uuid4()
    # clear feedback
    for key in list(st.session_state.keys()):
        if str(key).startswith("feedback_"):
            del st.session_state[key]


def format_feedback_kwargs(chat_history: list, n: int = -1, user_uuid: UUID = USER.uuid) -> dict:
    """Formats feedback kwarg dict based on a chat history."""
    return {
        "input": [msg["content"] for msg in chat_history],
        "chain": chat_history[0:n],
        "output": chat_history[n]["content"],
        "sources": chat_history[n]["sources"],
        "creator_user_uuid": user_uuid,
    }


if "chat_id" not in st.session_state:
    clear_chat()

if "messages" not in st.session_state:
    st.session_state.messages = {}

if st.session_state.chat_id not in st.session_state.messages:
    st.session_state.messages[st.session_state.chat_id] = []

# region Sidebar ====================

with st.sidebar:
    model_select = st.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        on_change=change_selected_model,
        key="model_select",
    )

    clear_chat_button = st.button("Clear Chat", on_click=clear_chat)

    st.download_button(
        label="Download Conversation",
        data=json.dumps(
            [x["content"] for x in st.session_state.messages[st.session_state.chat_id]],
            indent=4,
            ensure_ascii=False,
        ),
        file_name=(f"redboxai_conversation_{USER.uuid}" f"_{datetime.now().isoformat().replace('.', '_')}.json"),
    )

# region RAG chat ====================

# History

for i, chat_message in enumerate(st.session_state.messages[st.session_state.chat_id]):
    with st.chat_message(chat_message["role"], avatar=AVATAR_MAP[chat_message["role"]]):
        st.write(chat_message["content"])
        if chat_message["sources"]:
            st.markdown(
                render_document_citations(chat_message["sources"]),
                unsafe_allow_html=True,
            )

        if chat_message["role"] == "assistant":
            streamlit_feedback(
                **FEEDBACK_KWARGS,
                key=f"feedback_{i}",
                kwargs=format_feedback_kwargs(
                    chat_history=st.session_state.messages[st.session_state.chat_id],
                    n=i,
                ),
            )

# Input

if prompt := st.chat_input():
    st.chat_message("user", avatar=AVATAR_MAP["user"]).write(prompt)
    st.session_state.messages[st.session_state.chat_id].append({"role": "user", "content": prompt, "sources": None})

    with st.chat_message("assistant", avatar=AVATAR_MAP["assistant"]):
        response_with_source_stream = st.session_state.backend.rag_chat_stream(
            question=prompt, chat_uuid=st.session_state.chat_id
        )

        response_stream, sources_stream = itertools.tee(response_with_source_stream, 2)

        response = st.write_stream(get_stream_key(response_stream, "response"))
        sources = list(get_stream_key(sources_stream, "sources"))[0]
        st.markdown(render_document_citations(sources), unsafe_allow_html=True)

        st.session_state.messages[st.session_state.chat_id].append(
            {"role": "assistant", "content": response, "sources": sources}
        )

        streamlit_feedback(
            **FEEDBACK_KWARGS,
            key=f"feedback_{len(st.session_state.messages[st.session_state.chat_id]) - 1}",
            kwargs=format_feedback_kwargs(chat_history=st.session_state.messages[st.session_state.chat_id]),
        )
