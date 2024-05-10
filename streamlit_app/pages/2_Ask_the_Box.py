import json
from datetime import date, datetime

import streamlit as st
from streamlit_feedback import streamlit_feedback

from redbox.llm.prompts.core import CORE_REDBOX_PROMPT
from redbox.models import ChatMessage, ChatRequest, ChatResponse
from streamlit_app.utils import (
    StreamlitStreamHandler,
    change_selected_model,
    format_feedback_kwargs,
    init_session_state,
    response_to_message,
    submit_feedback,
)

st.set_page_config(page_title="Redbox Copilot - Ask the Box", page_icon="📮", layout="wide")

# region Global and session state variables, functions ====================

with st.spinner("Loading..."):
    ENV = init_session_state()
    AVATAR_MAP = {"human": "🧑‍💻", "ai": "📮", "user": "🧑‍💻", "assistant": "📮"}
    FEEDBACK_KWARGS = {
        "feedback_type": "thumbs",
        "optional_text_label": "What did you think of this response?",
        "on_submit": submit_feedback,
    }
    INITIAL_CHAT_PROMPT: list[ChatMessage | ChatResponse] = [
        ChatMessage(
            role="system",
            text=CORE_REDBOX_PROMPT.format(
                current_date=date.today().isoformat(),
                user_info=st.session_state.backend.get_user().str_llm(),
            ),
        ),
        ChatMessage(
            role="ai",
            text="Hi, I'm Redbox Copilot. How can I help you?",
        ),
    ]


def clear_chat() -> None:
    st.session_state["messages"] = INITIAL_CHAT_PROMPT
    # clear feedback
    for key in list(st.session_state.keys()):
        if str(key).startswith("feedback_"):
            del st.session_state[key]


if "messages" not in st.session_state:
    clear_chat()
    for key in list(st.session_state.keys()):
        if str(key).startswith("feedback_"):
            del st.session_state[key]

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
            [msg.model_dump_json() for msg in st.session_state.messages],
            indent=4,
            ensure_ascii=False,
        ),
        file_name=(
            f"redboxai_conversation_{st.session_state.backend.get_user().uuid}"
            f"_{datetime.now().isoformat().replace('.', '_')}.json"
        ),
    )

# region RAG chat ====================

# History

for i, msg in enumerate(st.session_state.messages):
    if msg.role == "system":
        continue

    with st.chat_message(msg.role, avatar=AVATAR_MAP[msg.role]):
        st.write(msg.text)
        if hasattr(msg, "sources"):
            st.markdown("\n".join([source.html for source in msg.sources]), unsafe_allow_html=True)

    if st.session_state.messages[i].role in ["ai", "assistant"] and i > 1:
        streamlit_feedback(
            **FEEDBACK_KWARGS,
            key=f"feedback_{i}",
            kwargs=format_feedback_kwargs(
                chat_history=st.session_state.messages, n=i, user_uuid=st.session_state.backend.get_user().uuid
            ),
        )

# Input

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", text=prompt))
    st.chat_message("user", avatar=AVATAR_MAP["user"]).write(prompt)

    with st.chat_message("assistant", avatar=AVATAR_MAP["assistant"]):
        response_stream_text = st.empty()

        chat_request = ChatRequest(message_history=st.session_state.messages)

        response_raw = st.session_state.backend.rag_chat(
            chat_request=chat_request,
            callbacks=[
                StreamlitStreamHandler(text_element=response_stream_text, initial_text=""),
                st.session_state.llm_logger_callback,
            ],
        )

        response = response_to_message(response=response_raw)

        response_stream_text.empty()
        response_stream_text.markdown(response.text, unsafe_allow_html=True)
        if hasattr(response, "sources"):
            st.markdown("\n".join([source.html for source in response.sources]), unsafe_allow_html=True)

    st.session_state.messages.append(response)

    streamlit_feedback(
        **FEEDBACK_KWARGS,
        key=f"feedback_{len(st.session_state.messages) - 1}",
        kwargs=format_feedback_kwargs(
            chat_history=st.session_state.messages, user_uuid=st.session_state.backend.get_user().uuid
        ),
    )
