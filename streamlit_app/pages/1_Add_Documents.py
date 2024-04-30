import pathlib
from uuid import UUID

import streamlit as st

from redbox.models import ContentType, File
from redbox.models.file import UploadFile
from streamlit_app.utils import init_session_state

st.set_page_config(page_title="Redbox Copilot - Add Documents", page_icon="📮", layout="wide")

ENV = init_session_state()

tags = st.session_state.backend.list_tags()

# Upload form


uploaded_files = st.file_uploader(
    "Upload your documents",
    accept_multiple_files=True,
    type=st.session_state.backend.get_supported_file_types(),
)

new_tag_uuid = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
no_tag_uuid = UUID("00000000-0000-0000-0000-000000000000")
tag_uuid_name_map_raw = {x.uuid: x.name for x in tags}
tag_uuid_name_map = {new_tag_uuid: "➕ New tag...", no_tag_uuid: "No associated tag", **tag_uuid_name_map_raw}

tag_selection = st.selectbox(
    "Add to tag:",
    options=tag_uuid_name_map.keys(),
    index=list(tag_uuid_name_map.keys()).index(new_tag_uuid),
    format_func=lambda x: tag_uuid_name_map[x],
)

new_tag = st.text_input("New tag name:")


# Create text input for user entry for new tag

submitted = st.button("Upload to Redbox Copilot")

if submitted and uploaded_files is not None:  # noqa: C901
    if tag_selection == new_tag_uuid:
        if not new_tag:
            st.error("Please enter a tag name")
            st.stop()
        elif new_tag in tag_uuid_name_map.values():
            st.error("Tag name already exists")
            st.stop()

    files: list[File] = []
    for file_index, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Uploading {uploaded_file.name}"):
            sanitised_name = uploaded_file.name
            sanitised_name = sanitised_name.replace("'", "_")

            file_type = pathlib.Path(sanitised_name).suffix

            file_to_upload = UploadFile(
                content_type=ContentType(file_type),
                filename=sanitised_name,
                creator_user_uuid=st.session_state.backend.get_user().uuid,
                file=uploaded_file,
            )

            file = st.session_state.backend.create_file(file_to_upload)
            files.append(file)

        st.toast(body=f"{file.name} Complete")

    # associate selected tag with the uploaded files
    if tag_selection != no_tag_uuid:
        if tag_selection == new_tag_uuid:
            tag_selection = st.session_state.backend.create_tag(name=new_tag).uuid

        st.session_state.backend.add_files_to_tag(file_uuids=[file.uuid for file in files], tag_uuid=tag_selection)
