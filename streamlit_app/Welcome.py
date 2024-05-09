import streamlit as st
from utils import init_session_state

from redbox.api import APIBackend
from redbox.models import Settings

st.set_page_config(
    page_title="Redbox Copilot",
    page_icon="📮",
)

ENV = init_session_state(backend=APIBackend(settings=Settings()))

st.write("# Redbox Copilot")


st.markdown(
    "### What can you do? \n\n"
    "* [Add Documents](/Add_Documents) by uploading them \n"
    "* [Summarise Documents](/Summarise_Documents) to extract key dates, "
    "people, actions and discussion for your principal \n"
    "* [Ask the Box](/Ask_the_Box) will answer questions about your box's content \n"
)
