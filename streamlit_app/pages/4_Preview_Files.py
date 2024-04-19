import uuid

import streamlit as st

from streamlit_app.utils import FilePreview, init_session_state

st.set_page_config(page_title="Redbox Copilot - Preview Files", page_icon="📮", layout="wide")

ENV = init_session_state()
file_preview = FilePreview()


if "file_uuid_to_name_map" not in st.session_state:
    st.session_state["file_uuid_to_name_map"] = {}


def refresh_files():
    st.session_state["file_uuid_to_name_map"] = {x.uuid: x.name for x in st.session_state.backend.list_files()}


def clear_params():
    st.query_params.clear()


st.title("Preview Files")

refresh_files()
url_params = st.query_params.to_dict()


select_index = 0
if "file_uuid" in url_params:
    select_index = st.session_state.file_uuid_to_name_map.keys().index(uuid.UUID(url_params["file_uuid"]))

file_select = st.selectbox(
    label="File",
    options=st.session_state.file_uuid_to_name_map.keys(),
    index=select_index,
    format_func=lambda x: st.session_state.file_uuid_to_name_map[x],
    on_change=clear_params,
)

col1, col2 = st.columns(2)
with col1:
    preview_file_button = st.button("🔍 Preview File")
with col2:
    delete_file_button = st.button("🗑️ Delete File")

if preview_file_button or "file_uuid" in url_params:
    file = st.session_state.backend.get_file(file_uuid=file_select)

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
    file = st.session_state.backend.delete_file(file_uuid=file_select)

    st.toast(f"Deleted file {file.name}", icon="🗑️")

    # Update the file list
    refresh_files()
