from datetime import datetime
from io import BytesIO
from typing import Optional
from uuid import UUID

import streamlit as st
from streamlit_feedback import streamlit_feedback

from redbox.export.docx import summary_tasks_to_docx
from redbox.llm.prompts.summary import SUMMARY_COMBINATION_TASK_PROMPT
from redbox.llm.summary import summary
from redbox.models import ChatMessage, ChatMessageSourced, File, SummaryTaskComplete
from streamlit_app.utils import (
    StreamlitStreamHandler,
    change_selected_model,
    init_session_state,
    response_to_message,
    slugify,
    submit_feedback,
)

st.set_page_config(page_title="Redbox Copilot - Ask the Box", page_icon="📮", layout="wide")

# region Global and session state variables, functions ====================

with st.spinner("Loading..."):
    ENV = init_session_state()
    FEEDBACK_KWARGS = {
        "feedback_type": "thumbs",
        "optional_text_label": "What did you think of this response?",
        "on_submit": submit_feedback,
    }
    MAX_TOKENS = 100_000
    URL_PARAMS = st.query_params.to_dict()
    TAGS = {tag.uuid: tag for tag in st.session_state.backend.list_tags()}
    FILES = {file.uuid: file for file in st.session_state.backend.list_files()}
    SUMMARY_TASKS = [
        summary.summary_task,
        summary.key_discussion_task,
        summary.key_actions_task,
        summary.key_people_task,
    ]


if "current_token_count" not in st.session_state:
    st.session_state.current_token_count = 0


if "submitted" not in st.session_state:
    st.session_state.submitted = False


if "summary" not in st.session_state:
    st.session_state.summary = []


if "summary_of_summaries_mode" not in st.session_state:
    st.session_state.summary_of_summaries_mode = False


def update_token_budget_tracker():
    current_token_count = 0

    for selected_file_uuid in st.session_state.selected_files:
        current_token_count += st.session_state.backend.get_file_token_count(file_uuid=selected_file_uuid)

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


def summary_to_markdown(tasks: list[SummaryTaskComplete], title: Optional[str] = None) -> str:
    out = ""

    if title is None:
        out += "# Redbox Copilot \n\n"
    else:
        out += f"# {title} \n\n"

    for task in tasks:
        out += f"## {task.title} \n\n"
        out += f"{task.response_text} \n\n"
        out += "### Sources \n\n"

        source_files = st.session_state.backend.get_files(file_uuids=task.file_uuids)
        source_file_bullets = " \n".join([f"* {file.key}" for file in source_files])

        out += f"{source_file_bullets} \n\n"

    out += "---------------------------------------------------\n"
    out += "This summary is AI Generated and may be inaccurate."

    return out


def summary_to_docx(tasks: list[SummaryTaskComplete], title: Optional[str] = None) -> BytesIO:
    reference_files: set[File] = set()
    for task in tasks:
        reference_files.update(st.session_state.backend.get_files(file_uuids=task.file_uuids))

    if len(reference_files) == 1:
        title = slugify(list(reference_files)[0].key)

    document = summary_tasks_to_docx(tasks=tasks, reference_files=reference_files, title=title)

    bytes_document = BytesIO()
    document.save(bytes_document)
    bytes_document.seek(0)

    return bytes_document


def delete_summary(file_uuids: list[UUID]) -> None:
    _ = st.session_state.backend.delete_summary(file_uuids=file_uuids)
    st.session_state.summary = []
    st.session_state.submitted = False
    st.query_params.clear()


# region Sidebar ====================


with st.sidebar:
    model_select = st.selectbox(
        "Select Model",
        options=st.session_state.available_models,
        on_change=change_selected_model,
        key="model_select",
    )

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

    tag = TAGS.get(getattr(st.session_state, "selected_tag", None))
    tag_name = tag.name if tag is not None else None
    tag_file_prefix = f"{tag.name}_" if tag is not None else ""
    summary_file_name_root = f"{tag_file_prefix}{datetime.now().isoformat()}_summary"

    st.download_button(
        label="Download DOCX",
        data=summary_to_docx(tasks=st.session_state.summary, title=tag_name),
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_name=f"{summary_file_name_root}.docx",
    )

    st.download_button(
        label="Download TXT",
        data=summary_to_markdown(tasks=st.session_state.summary, title=tag_name),
        mime="text/plain",
        file_name=f"{summary_file_name_root}.txt",
    )

    delete_summary_button = st.button(
        label="Delete Summary",
        on_click=delete_summary,
        kwargs={"file_uuids": getattr(st.session_state, "selected_files", [])},
    )


# region File select ====================

st.title("Summarise Documents")

tag_index = None

if URL_PARAMS.get("tag_uuid") is not None:
    tag = st.session_state.backend.get_tag(tag_uuid=URL_PARAMS.get("tag_uuid"))
    tag_index = list(TAGS.keys()).index(tag.uuid)

tag_select = st.selectbox(
    label="Tags",
    options=TAGS.keys(),
    index=tag_index,
    format_func=lambda tag: TAGS[tag].name,
    key="selected_tag",
)

summary_file_select = st.multiselect(
    label="Files",
    options=FILES.keys(),
    default=TAGS[tag_select].files if tag_select is not None else [],
    on_change=clear_params,
    format_func=lambda file: FILES[file].key,
    key="selected_files",
)

update_token_budget_tracker()

submitted = st.button("Redbox Copilot Summary")

# Using this state trick to allow post gen download without reload.
if submitted:
    if summary_file_select:
        st.session_state.submitted = True
    else:
        st.warning("Please select document(s)")
        unsubmit_session_state()


# region Summarise documents ====================


if st.session_state.submitted:
    saved_summary = st.session_state.backend.get_summary(file_uuids=summary_file_select)
    if saved_summary is not None:
        st.info("Loading cached summary")
        st.session_state.summary = saved_summary.tasks

    # Render summary
    rendered_tasks: list[str] = []
    for task in st.session_state.summary:
        st.subheader(task.title, divider=True)
        st.markdown(task.response_text, unsafe_allow_html=True)
        if hasattr(task, "sources"):
            st.markdown("\n".join([source.html for source in task.sources]), unsafe_allow_html=True)

        streamlit_feedback(
            **FEEDBACK_KWARGS,
            key=f"feedback_{task.id}",
            kwargs={
                "input": ChatMessageSourced(
                    text=task.prompt_template.pretty_repr(), sources=task.sources, role="system"
                ),
                "output": ChatMessage(text=task.response_text, role="ai"),
                "creator_user_uuid": st.session_state.backend.get_user().uuid,
            },
        )

        rendered_tasks.append(task.id)

    # Run summary
    for task in [task for task in SUMMARY_TASKS if task.id not in rendered_tasks]:
        response_stream_header = st.subheader(task.title, divider=True)
        with st.status(
            f"Generating {task.title}",
            expanded=not st.session_state.summary_of_summaries_mode,
            state="running",
        ):
            response_stream_text = st.empty()

            if st.session_state.summary_of_summaries_mode:
                response_raw = st.session_state.backend.map_reduce_summary(
                    map_prompt=task.prompt_template,
                    reduce_prompt=SUMMARY_COMBINATION_TASK_PROMPT,
                    file_uuids=summary_file_select,
                    callbacks=[
                        StreamlitStreamHandler(
                            text_element=response_stream_text,
                            initial_text="",
                        ),
                        st.session_state.llm_logger_callback,
                    ],
                )
            else:
                response_raw = st.session_state.backend.stuff_doc_summary(
                    summary=task.prompt_template,
                    file_uuids=summary_file_select,
                    callbacks=[
                        StreamlitStreamHandler(
                            text_element=response_stream_text,
                            initial_text="",
                        ),
                        st.session_state.llm_logger_callback,
                    ],
                )

            response = response_to_message(response=response_raw)

            response_stream_header.empty()
            response_stream_text.empty()
            response_stream_text.markdown(response.text, unsafe_allow_html=True)
            if hasattr(response, "sources"):
                st.markdown("\n".join([source.html for source in response.sources]), unsafe_allow_html=True)

        complete = SummaryTaskComplete(
            id=task.id,
            title=task.title,
            prompt_template=task.prompt_template,
            file_uuids=summary_file_select,
            response_text=response.text,
            sources=getattr(response, "sources", None),
            creator_user_uuid=st.session_state.backend.get_user().uuid,
        )
        st.session_state.summary.append(complete)
        st.rerun()

    # Save summary
    if saved_summary is None:
        st.session_state.backend.create_summary(file_uuids=summary_file_select, tasks=st.session_state.summary)
