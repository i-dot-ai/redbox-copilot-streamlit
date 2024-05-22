import time
from pathlib import Path
from uuid import UUID

import streamlit as st

from redbox.models import ContentType, File, UploadFile
from streamlit_app.utils import FilePreview, init_session_state

st.set_page_config(page_title="Redbox - Files", page_icon="📮", layout="wide")

# region Global and session state variables, functions ====================

with st.spinner("Loading..."):
    ENV = init_session_state()
    TAGS = st.session_state.backend.list_tags()
    FILES = st.session_state.backend.list_files()
    URL_PARAMS = st.query_params.to_dict()
    FILE_PREVIEW = FilePreview(backend=st.session_state.backend)


@st.experimental_dialog("File preview", width="large")
def preview_modal(file):
    content_type = Path(file.key).suffix
    if content_type in FILE_PREVIEW.render_methods:
        if (content_type == ".pdf") & ("page_number" in URL_PARAMS):
            page_number_raw = URL_PARAMS["page_number"]
            if page_number_raw[0] == "[":
                page_numbers = page_number_raw[1:-1].split(r",")
                page_number = min([int(p) for p in page_numbers])
            else:
                page_number = int(page_number_raw)
            FILE_PREVIEW._render_pdf(file, page_number=page_number)
        else:
            FILE_PREVIEW.st_render(file, content_type=content_type)
    else:
        st.warning(f"File rendering not yet supported for {content_type}")


# region Upload form ====================

with st.form("Upload", clear_on_submit=True):
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=st.session_state.backend.get_supported_file_types(),
    )

    new_tag_uuid = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    no_tag_uuid = UUID("00000000-0000-0000-0000-000000000000")
    tag_uuid_name_map_raw = {x.uuid: x.name for x in TAGS}
    tag_uuid_name_map = {new_tag_uuid: "➕ New tag...", no_tag_uuid: "No associated tag", **tag_uuid_name_map_raw}

    col_add, col_new = st.columns(2)

    tag_selection = col_add.selectbox(
        "Add to tag:",
        options=tag_uuid_name_map.keys(),
        index=list(tag_uuid_name_map.keys()).index(new_tag_uuid),
        format_func=lambda x: tag_uuid_name_map[x],
    )

    new_tag = col_new.text_input("New tag name:")

    submitted = st.form_submit_button("Upload to Redbox", type="primary")

    if submitted and uploaded_files is not None:  # noqa: C901
        if tag_selection == new_tag_uuid:
            if not new_tag:
                st.error("Please enter a tag name")
                st.stop()
            elif new_tag in tag_uuid_name_map.values():
                st.error("Tag name already exists")
                st.stop()

        files: list[File] = []
        for uploaded_file in uploaded_files:
            with st.spinner(f"Uploading {uploaded_file.name}"):
                sanitised_name = uploaded_file.name
                sanitised_name = sanitised_name.replace("'", "_")

                file_type = Path(sanitised_name).suffix

                file_to_upload = UploadFile(
                    content_type=ContentType(file_type),
                    filename=sanitised_name,
                    creator_user_uuid=st.session_state.backend.get_user().uuid,
                    file=uploaded_file,
                )

                file = st.session_state.backend.create_file(file_to_upload)
                files.append(file)

            st.toast(body=f"{file.key} added to processing queue")

        # associate selected tag with the uploaded files
        if tag_selection != no_tag_uuid:
            if tag_selection == new_tag_uuid:
                tag_selection = st.session_state.backend.create_tag(name=new_tag).uuid

            st.session_state.backend.add_files_to_tag(file_uuids=[file.uuid for file in files], tag_uuid=tag_selection)

# region File management ====================

st.divider()

for i, file in enumerate(FILES):
    col_name, col_status, col_preview, col_delete = st.columns((2, 1, 1, 1))
    status = st.session_state.backend.get_file_status(file_uuid=file.uuid)

    col_name.write(file.key)
    col_status.write(status.processing_status.title())
    col_preview.button("🔍 Preview File", key=f"preview_{i}")
    col_delete.button("🗑️ Delete File", key=f"delete_{i}", type="primary")

    if st.session_state[f"preview_{i}"] or "file_uuid" in URL_PARAMS:
        preview_modal(file)

    if st.session_state[f"delete_{i}"]:
        file = st.session_state.backend.delete_file(file_uuid=file.uuid)
        time.sleep(1)
        st.toast(f"Deleted file {file.key}", icon="🗑️")
        st.rerun()

st.divider()

if st.button("↻ Refresh file status"):
    st.rerun()
