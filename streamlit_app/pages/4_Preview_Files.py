import uuid

import streamlit as st

from streamlit_app.utils import FilePreview, init_session_state

st.set_page_config(page_title="Redbox Copilot - Preview Files", page_icon="📮", layout="wide")

ENV = init_session_state()
file_preview = FilePreview()


st.title("Preview Files")

st.session_state["files"] = []
st.session_state["file_uuid_map"] = {}
st.session_state["file_uuid_to_name_map"] = {}


def refresh_files():
    st.session_state["files"] = st.session_state.storage_handler.read_all_items(model_type="File")
    st.session_state["file_uuid_map"] = {x.uuid: x for x in st.session_state["files"]}
    st.session_state["file_uuid_to_name_map"] = {x.uuid: x.name for x in st.session_state["files"]}


refresh_files()
url_params = st.query_params.to_dict()


def clear_params():
    st.query_params.clear()


if "file_uuid" in url_params:
    file_uuid_string_list = [str(file_uuid) for file_uuid in st.session_state.file_uuid_to_name_map.keys()]
    file_select = st.selectbox(
        label="File",
        options=file_uuid_string_list,
        index=file_uuid_string_list.index(url_params["file_uuid"]),
        on_change=clear_params,
        format_func=lambda x: st.session_state.file_uuid_to_name_map[uuid.UUID(x)],
    )
else:
    file_select = st.selectbox(
        label="File",
        index=0,
        options=list(st.session_state.file_uuid_to_name_map.keys()),
        format_func=lambda x: st.session_state.file_uuid_to_name_map[x],
    )
col1, col2 = st.columns(2)
with col1:
    preview_file_button = st.button("🔍 Preview File")
with col2:
    delete_file_button = st.button("🗑️ Delete File")

if preview_file_button or "file_uuid" in url_params:
    file = st.session_state.file_uuid_map[uuid.UUID(file_select)]

    with st.expander("File Metadata"):
        st.markdown(f"**Name:** `{file.name}`")
        st.markdown(f"**UUID:** `{file.uuid}`")
        st.markdown(f"**Type:** `{file.content_type}`")
        st.markdown(f"**Token Count:** `{file.token_count}`")
        st.markdown(f"**Text Hash:** `{file.text_hash}`")
        st.markdown(f"**Creator UUID:** `{file.creator_user_uuid}`")

    if file.content_type in file_preview.render_methods:
        if (file.content_type == ".pdf") & ("page_number" in url_params):
            page_number_raw = url_params["page_number"]
            if page_number_raw[0] == "[":
                page_numbers = page_number_raw[1:-1].split(r",")
                page_number = min([int(p) for p in page_numbers])
            else:
                page_number = int(page_number_raw)
            file_preview._render_pdf(file, page_number=page_number)
        else:
            file_preview.st_render(file)
    else:
        st.warning(f"File rendering not yet supported for {file.content_type}")

if delete_file_button:
    file = st.session_state.file_uuid_map[file_select]
    # Update Tag.files to remove all references to this file
    tags = st.session_state.storage_handler.read_all_items("Tag")
    for tag in tags:
        if str(file.uuid) in tag.files:
            tag.files.remove(str(file.uuid))
            if len(tag.files) >= 1:
                st.session_state.storage_handler.update_item(tag)
            else:
                st.session_state.storage_handler.delete_item(tag)
                st.toast(
                    f"Deleted tag {tag.name} as it was empty",
                    icon="🗑️",
                )

    # Delete the file from Uploads
    st.session_state.s3_client.delete_object(Bucket=st.session_state.BUCKET_NAME, Key=file.name)

    # Delete the file from the DB
    st.session_state.storage_handler.delete_item(file)

    st.toast(f"Deleted file {file.name}", icon="🗑️")

    # Update the file list
    refresh_files()
