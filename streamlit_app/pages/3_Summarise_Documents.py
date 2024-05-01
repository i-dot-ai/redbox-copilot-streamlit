import datetime
import uuid
from io import BytesIO

import streamlit as st
from streamlit_feedback import streamlit_feedback

from redbox.export.docx import summary_complete_to_docx
from redbox.models import SummaryComplete, SummaryTaskComplete, File
from streamlit_app.utils import (
    StreamlitStreamHandler,
    hash_list_of_files,
    init_session_state,
    load_llm_handler,
    replace_doc_ref,
    submit_feedback,
)

# region ===== PAGE SETUP =====

st.set_page_config(
    page_title="Redbox Copilot - Summarise Documents",
    page_icon="📮",
    layout="wide",
)

st.title("Summarise Documents")

with st.spinner("Loading..."):
    ENV = init_session_state()

# Model selector


def change_selected_model():
    load_llm_handler(ENV, update=True)
    st.write(st.session_state.llm)


model_select = st.sidebar.selectbox(
    "Select Model",
    options=st.session_state.available_models,
    on_change=change_selected_model,
    key="model_select",
)


MAX_TOKENS = 100_000
if "current_token_count" not in st.session_state:
    st.session_state.current_token_count = 0

token_budget_ratio = float(st.session_state.current_token_count) / MAX_TOKENS
token_budget_tracker = st.sidebar.progress(
    value=(token_budget_ratio if token_budget_ratio <= 1 else 1),
)
token_budget_desc = st.sidebar.caption(body=f"Word Budget: {st.session_state.current_token_count}/{MAX_TOKENS}")


def on_summary_of_summaries_mode_change():
    st.session_state.summary_of_summaries_mode = not (st.session_state.summary_of_summaries_mode)
    if st.session_state.summary_of_summaries_mode:
        st.sidebar.info("Will summarise each document individually and combine them.")
    unsubmit_session_state()


summary_of_summaries_mode = st.sidebar.toggle(
    "Summary of Summaries",
    value=st.session_state.summary_of_summaries_mode,
    on_change=on_summary_of_summaries_mode_change,
)

feedback_kwargs = {
    "feedback_type": "thumbs",
    "optional_text_label": "What did you think of this response?",
    "on_submit": submit_feedback,
}


# endregion

# summary_file_select = st.empty()


def update_token_budget_tracker():
    current_token_count = 0

    for selected_file_uuid in st.session_state.selected_files:
        selected_file = parsed_files_uuid_map[selected_file_uuid]
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


if "submitted" not in st.session_state:
    st.session_state.submitted = False


# region ===== URL PARAM INJECTION =====
url_params = st.query_params.to_dict()

tags = st.session_state.storage_handler.read_all_items("Tag")
tags_by_uuid_map = {x.uuid: x for x in tags}
tag = None


parsed_files = st.session_state.storage_handler.read_all_items("File")
parsed_files_uuid_map = {x.uuid: x for x in parsed_files}

tag_select = st.selectbox(
    label="Tags",
    options=tags_by_uuid_map.keys(),
    index=None,
    format_func=lambda x: tags_by_uuid_map[x].name,
)

if tag_select is not None:
    url_params["tag_title"] = str(tag_select)

files_from_url = []
if "tag_title" in url_params:
    tag_title = url_params["tag_title"]
    tag = st.session_state.storage_handler.read_item(item_uuid=uuid.UUID(tag_title), model_type="Tag")

    files_from_url = [uuid.UUID(x) for x in tag.files if isinstance(x, str)]
    files_from_url = [x for x in files_from_url if x in parsed_files_uuid_map.keys()]

summary_file_select = st.multiselect(
    label="Files",
    options=parsed_files_uuid_map.keys(),
    default=files_from_url,
    on_change=clear_params,
    format_func=lambda x: parsed_files_uuid_map[x].name,
    key="selected_files",
)

update_token_budget_tracker()

# endregion

submitted = st.button("Redbox Copilot Summary")

# Using this state trick to allow post gen download without reload.
if submitted:
    if summary_file_select:
        st.session_state.submitted = True
    else:
        st.warning("Please select document(s)")
        unsubmit_session_state()

files = []
for file in summary_file_select:
    file_to_add = parsed_files_uuid_map[file]
    file_to_add_chunks = st.session_state.storage_handler.get_file_chunks(file_to_add.uuid)

    file_to_add_with_text = File(
        **{k: v for k, v in file_to_add.dict().items() if k != "text"},
        text="\n".join([chunk.text for chunk in file_to_add_chunks]),
    )

    files.append(file_to_add_with_text)

if len(files) == 0:
    st.stop()
SELECTED_FILE_HASH = hash_list_of_files(files)

summary_completed = st.session_state.storage_handler.read_all_items(model_type="SummaryComplete")
summary_completed_by_hash = {x.file_hash: x for x in summary_completed}


# RENDER SUMMARY
if st.session_state.submitted:
    if SELECTED_FILE_HASH in summary_completed_by_hash:
        st.info("Loading cached summary")
        cached_complete = summary_completed_by_hash[SELECTED_FILE_HASH]
        st.session_state.summary = cached_complete.tasks

    for completed_task in st.session_state.summary:
        st.subheader(completed_task.title, divider=True)
        st.markdown(completed_task.processed, unsafe_allow_html=True)
        streamlit_feedback(
            **feedback_kwargs,
            key=f"feedback_{completed_task.id}",
            kwargs={
                "input": [f.to_document().page_content for f in files],
                "chain": completed_task.chain,
                "output": completed_task.raw,
                "creator_user_uuid": st.session_state.backend.get_user().uuid,
            },
        )

    if SELECTED_FILE_HASH not in summary_completed_by_hash:
        # RUN SUMMARY
        summary_model = st.session_state.llm_handler.get_summary_tasks(files=files, file_hash=SELECTED_FILE_HASH)
        finished_tasks = [t.id for t in st.session_state.summary]
        for task in summary_model.tasks:
            if task.id not in finished_tasks:
                response_stream_header = st.subheader(task.title, divider=True)
                with st.status(
                    f"Generating {task.title}",
                    expanded=not st.session_state.summary_of_summaries_mode,
                    state="running",
                ):
                    response_stream_text = st.empty()
                    with response_stream_text:
                        (
                            response,
                            chain,
                        ) = st.session_state.llm_handler.run_summary_task(
                            summary=summary_model,
                            task=task,
                            user_info=st.session_state.backend.get_user().str_llm(),
                            callbacks=[
                                StreamlitStreamHandler(
                                    text_element=response_stream_text,
                                    initial_text="",
                                ),
                                st.session_state.llm_logger_callback,
                            ],
                            map_reduce=st.session_state.summary_of_summaries_mode,
                        )
                        response_final_markdown = replace_doc_ref(response, files)

                        response_stream_header.empty()
                        response_stream_text.empty()

                complete = SummaryTaskComplete(
                    id=task.id,
                    title=task.title,
                    chain=chain,
                    file_hash=summary_model.file_hash,
                    raw=response,
                    processed=response_final_markdown,
                    creator_user_uuid=st.session_state.backend.get_user().uuid,
                )
                st.session_state.summary.append(complete)
                st.rerun()

        summary_complete = SummaryComplete(
            file_hash=summary_model.file_hash,
            file_uuids=[str(f.uuid) for f in files],
            tasks=st.session_state.summary,
            creator_user_uuid=st.session_state.backend.get_user().uuid,
        )

        st.session_state.storage_handler.write_item(item=summary_complete)
        summary_completed_by_hash[SELECTED_FILE_HASH] = summary_complete

    def summary_to_markdown():
        out = ""
        for completed_task in st.session_state.summary:
            out += "## " + completed_task.title + "\n\n"
            out += completed_task.processed + "\n\n"
        out += "---------------------------------------------------\n"
        out += "This summary is AI Generated and may be inaccurate."
        return out

    def summary_to_docx():
        if tag is not None:
            document = summary_complete_to_docx(
                summary_complete=summary_completed_by_hash[SELECTED_FILE_HASH],
                files=files,
                title=tag.name,
            )
        elif len(files) == 1:
            sanitised_file_name = files[0].name.replace("_", " ").replace("-", " ")
            # remove file extension
            sanitised_file_name = sanitised_file_name[: sanitised_file_name.rfind(".")]
            sanitised_file_name = sanitised_file_name.strip()

            # replace any multiple spaces with single space
            sanitised_file_name = " ".join(sanitised_file_name.split())

            document = summary_complete_to_docx(
                summary_complete=summary_completed_by_hash[SELECTED_FILE_HASH],
                files=files,
                title=sanitised_file_name,
            )
        else:
            document = summary_complete_to_docx(
                summary_complete=summary_completed_by_hash[SELECTED_FILE_HASH],
                files=files,
            )
        bytes_document = BytesIO()
        document.save(bytes_document)
        bytes_document.seek(0)
        return bytes_document

    if tag is not None:
        summary_file_name_root = f"{tag.name}_{datetime.datetime.now().isoformat()}_summary"
    else:
        summary_file_name_root = f"{datetime.datetime.now().isoformat()}_summary"

    st.sidebar.download_button(
        label="Download DOCX",
        data=summary_to_docx(),
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        file_name=f"{summary_file_name_root}.docx",
    )

    st.sidebar.download_button(
        label="Download TXT",
        data=summary_to_markdown(),
        mime="text/plain",
        file_name=f"{summary_file_name_root}.txt",
    )

    def delete_summary():
        summary_completed_to_delete = summary_completed_by_hash[SELECTED_FILE_HASH]
        st.session_state.storage_handler.delete_item(summary_completed_to_delete)
        del summary_completed_by_hash[SELECTED_FILE_HASH]

        st.session_state.summary = []
        st.session_state.submitted = False
        st.query_params.clear()

    delete_summary_button = st.sidebar.button(
        label="Delete Summary",
        on_click=delete_summary,
    )
