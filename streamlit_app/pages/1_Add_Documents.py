import pathlib
from uuid import UUID

import streamlit as st

from redbox.models import Collection, ContentType
from redbox.models.file import UploadFile
from redbox.parsing.file_chunker import FileChunker
from streamlit_app.utils import init_session_state

st.set_page_config(page_title="Redbox Copilot - Add Documents", page_icon="📮", layout="wide")

ENV = init_session_state()

file_chunker = FileChunker(embedding_model=st.session_state.embedding_model)

collections = st.session_state.storage_handler.read_all_items(model_type="Collection")

# Upload form


uploaded_files = st.file_uploader(
    "Upload your documents",
    accept_multiple_files=True,
    type=file_chunker.supported_file_types,
)

new_collection_str = "➕ New collection..."
no_collection_str = " No associated collection"
collection_uuid_name_map = {x.uuid: x.name for x in collections}
collection_uuid_name_map[new_collection_str] = new_collection_str
collection_uuid_name_map[no_collection_str] = no_collection_str

collection_selection = st.selectbox(
    "Add to collection:",
    options=collection_uuid_name_map.keys(),
    index=list(collection_uuid_name_map.keys()).index(new_collection_str),
    format_func=lambda x: collection_uuid_name_map[x],
)

new_collection = st.text_input("New collection name:")


# Create text input for user entry for new collection

submitted = st.button("Upload to Redbox Copilot collection")


if submitted and uploaded_files is not None:  # noqa: C901
    if collection_selection == new_collection_str:
        if not new_collection:
            st.error("Please enter a collection name")
            st.stop()
        elif new_collection in collection_uuid_name_map.values():
            st.error("Collection name already exists")
            st.stop()

    # associate selected collection with the uploaded files
    if collection_selection == new_collection_str:
        collection_obj = Collection(
            name=new_collection,
            creator_user_uuid=UUID(st.session_state.user_uuid),
        )
    elif collection_selection == no_collection_str:
        collection_obj = Collection(
            name="",
            creator_user_uuid=UUID(st.session_state.user_uuid),
        )
    else:
        collection_obj = st.session_state.storage_handler.read_item(
            item_uuid=collection_selection, model_type="Collection"
        )

    for file_index, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f"Uploading {uploaded_file.name}"):
            sanitised_name = uploaded_file.name
            sanitised_name = sanitised_name.replace("'", "_")

            file_type = pathlib.Path(sanitised_name).suffix

            file_to_upload = UploadFile(
                content_type=ContentType(file_type),
                filename=sanitised_name,
                creator_user_uuid=UUID(st.session_state.user_uuid),
                file=uploaded_file,
            )

            file = st.session_state.backend.add_file(file_to_upload)

        st.toast(body=f"{file.name} Complete")

        collection_obj.files.append(str(file.uuid))

    if collection_obj.name and (collection_obj.name != "none"):
        st.session_state.storage_handler.write_item(item=collection_obj)
