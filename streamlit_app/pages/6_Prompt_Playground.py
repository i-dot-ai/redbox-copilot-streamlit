import streamlit as st
import itertools
from uuid import uuid4
import json
from datetime import datetime

from redbox.llm.prompts.core import _core_redbox_prompt
from redbox.llm.prompts.chat import _with_sources_template
from streamlit_app.utils import (
    init_session_state,
    submit_feedback,
    change_selected_model,
    render_document_citations,
    get_stream_key,
    LOG,
)

st.set_page_config(
    page_title="Redbox Copilot - Prompt Playground",
    page_icon="📮",
    layout="wide",
)

# region Global and session state variables, functions ====================

with st.spinner("Loading..."):
    ENV = init_session_state()
    USER = st.session_state.backend.get_user()
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
    MAX_TOKENS = 100_000
    URL_PARAMS = st.query_params.to_dict()
    TAGS = {tag.uuid: tag for tag in st.session_state.backend.list_tags()}
    FILES = {file.uuid: file for file in st.session_state.backend.list_files()}

if "current_token_count" not in st.session_state:
    st.session_state.current_token_count = 0

if "submitted" not in st.session_state:
    st.session_state.submitted = False

if "prompt_list" not in st.session_state:
    st.session_state.prompt_list = []


def update_token_budget_tracker():
    current_token_count = 0

    for selected_file_uuid in st.session_state.selected_files:
        selected_file = FILES[selected_file_uuid]
        LOG.info(selected_file)
        current_token_count += selected_file.token_count

    if current_token_count > MAX_TOKENS:
        if not st.session_state.summary_of_summaries_mode:
            st.session_state.summary_of_summaries_mode = True
            st.toast(
                "Summary of Summaries mode enabled due to token budget",
                icon="⚠️",
            )

    st.session_state.current_token_count = current_token_count


def clear_params():
    st.query_params.clear()
    unsubmit_session_state()


def unsubmit_session_state():
    update_token_budget_tracker()
    st.session_state.summary = []
    st.session_state.submitted = False


def on_summary_of_summaries_mode_change():
    st.session_state.summary_of_summaries_mode = not (st.session_state.summary_of_summaries_mode)
    if st.session_state.summary_of_summaries_mode:
        st.sidebar.info("Will summarise each document individually and combine them.")
    unsubmit_session_state()


def gen_default_prompt(n: int, chat_mode: bool) -> tuple[str, str]:
    if chat_mode:
        match n:
            case 0:
                return ("system", _core_redbox_prompt)
            case 1:
                return ("placeholder", "{history}")
            case 2:
                return ("human", _with_sources_template)
            case _ if n % 2:
                return ("ai", "")
            case _:
                return ("human", "")
    else:
        match n:
            case 0:
                return ("system", _core_redbox_prompt)
            case 1:
                return ("human", _with_sources_template)
            case _ if n % 2:
                return ("human", "")
            case _:
                return ("ai", "")


def clear_chat() -> None:
    st.session_state.chat_id = uuid4()
    # clear feedback
    for key in list(st.session_state.keys()):
        if str(key).startswith("feedback_"):
            del st.session_state[key]


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

    chat_mode = st.toggle("Chat mode", value=False)

    if chat_mode:
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
    else:
        token_budget_ratio = float(st.session_state.current_token_count) / MAX_TOKENS
        token_budget_tracker = st.progress(
            value=(token_budget_ratio if token_budget_ratio <= 1 else 1),
        )
        token_budget_desc = st.caption(body=f"Word Budget: {st.session_state.current_token_count}/{MAX_TOKENS}")

        summary_of_summaries_mode = st.toggle(
            "Summary of Summaries",
            value=st.session_state.summary_of_summaries_mode,
            on_change=on_summary_of_summaries_mode_change,
        )

# region Prompt creation ====================

st.title("Prompt Playground")
st.markdown("Use this page to craft prompts we can use to make Redbox better.")

with st.expander("Files", expanded=False):
    tag_index = None

    if URL_PARAMS.get("tag_uuid") is not None:
        tag = st.session_state.backend.get_tag(tag_uuid=URL_PARAMS.get("tag_uuid"))
        tag_index = list(TAGS.keys()).index(tag.uuid)

    tag_select = st.selectbox(
        label="Tags",
        options=TAGS.keys(),
        index=tag_index,
        format_func=lambda tag: TAGS[tag].name,
    )

    summary_file_select = st.multiselect(
        label="Files",
        options=FILES.keys(),
        default=TAGS[tag_select].files if tag_select is not None else [],
        on_change=clear_params,
        format_func=lambda file: FILES[file].name,
        key="selected_files",
    )

update_token_budget_tracker()

with st.expander("Prompts", expanded=False):
    required_fields = {
        "{current_date}",
        "{user_info}",
        "{summaries}",
        "{question}",
        "{history}",
    }
    roles = {"system", "placeholder", "human", "ai"}

    if chat_mode:
        required_fields.add("{history}")
        roles.add("{placeholder}")
    else:
        required_fields.discard("{history}")
        roles.discard("{placeholder}")

    prompt_count = st.number_input("Prompt count", min_value=0, value=3)
    current_prompt_count = len(st.session_state.prompt_list)

    if current_prompt_count > prompt_count:
        for i in range(prompt_count, current_prompt_count):
            del st.session_state[f"role_{i}"]
            del st.session_state[f"text_{i}"]
        st.session_state.prompt_list = st.session_state.prompt_list[:prompt_count]
    elif current_prompt_count < prompt_count:
        for i in range(current_prompt_count, prompt_count):
            st.session_state.prompt_list.append(gen_default_prompt(n=i, chat_mode=chat_mode))

    st.divider()

    altered_prompt_list: list[tuple[str, str]] = []
    role_list = list(roles)
    for i, prompt in enumerate(st.session_state.prompt_list):
        role_key = f"role_{i}"
        text_key = f"text_{i}"

        st.selectbox(
            "Role",
            options=role_list,
            index=role_list.index(prompt[0]),
            key=role_key,
        )
        text = st.text_area("System", value=prompt[1], height=200, key=text_key)
        altered_prompt_list.append((st.session_state[role_key], st.session_state[text_key]))

    st.session_state.prompt_list = altered_prompt_list

    field_warnings = []
    for field in required_fields:
        if not any(field in prompt[1] for prompt in st.session_state.prompt_list):
            field_warnings.append(field)

    if len(field_warnings) > 0:
        st.error(f"Prompts must contain: {', '.join(field_warnings)}")

# region Chat mode ====================

if chat_mode:
    # History

    for i, chat_message in enumerate(st.session_state.messages[st.session_state.chat_id]):
        with st.chat_message(chat_message["role"], avatar=AVATAR_MAP[chat_message["role"]]):
            st.write(chat_message["content"])
            if chat_message["sources"]:
                st.markdown(
                    render_document_citations(chat_message["sources"]),
                    unsafe_allow_html=True,
                )

    # Input

    if prompt := st.chat_input():
        st.chat_message("user", avatar=AVATAR_MAP["user"]).write(prompt)
        st.session_state.messages[st.session_state.chat_id].append({"role": "user", "content": prompt, "sources": None})

        with st.chat_message("assistant", avatar=AVATAR_MAP["assistant"]):
            response_with_source_stream = st.session_state.backend.rag_chat_stream(
                question=prompt,
                chat_uuid=st.session_state.chat_id,
                init_messages=st.session_state.prompt_list,
            )

            response_stream, sources_stream = itertools.tee(response_with_source_stream, 2)

            response = st.write_stream(get_stream_key(response_stream, "response"))
            sources = list(get_stream_key(sources_stream, "sources"))[0]
            st.markdown(render_document_citations(sources), unsafe_allow_html=True)

            st.session_state.messages[st.session_state.chat_id].append(
                {"role": "assistant", "content": response, "sources": sources}
            )

# region Summary mode ====================

else:
    submitted = st.button("Redbox Copilot Summary")

    # Using this state trick to allow post gen download without reload.
    if submitted:
        if summary_file_select:
            st.session_state.submitted = True
        else:
            st.warning("Please select document(s)")
            unsubmit_session_state()

    st.write(summary_file_select)
