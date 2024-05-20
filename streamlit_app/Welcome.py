import streamlit as st
from utils import init_session_state

st.set_page_config(
    page_title="Redbox Copilot",
    page_icon="📮",
)

ENV = init_session_state()

st.write("# Redbox Copilot")


st.markdown(
    "### What can you do? \n\n"
    "* [Add Documents](/Add_Documents) by uploading them \n"
    "* [Summarise Documents](/Summarise_Documents) to extract key dates, "
    "people, actions and discussion for your principal \n"
    "* [Ask the Box](/Ask_the_Box) will answer questions about your box's content \n"
)

files = st.session_state.backend.list_files()
chunks = st.session_state.backend.get_file_chunks(files[0].uuid)
st.write(chunks)
