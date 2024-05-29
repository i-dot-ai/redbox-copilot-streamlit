import streamlit as st
from utils import init_session_state

st.set_page_config(
    page_title="Redbox",
    page_icon="📮",
)

ENV = init_session_state()

st.title("Redbox")

st.markdown(
    "### What can you do? \n\n"
    "* [Add documents](/Documents) by uploading them \n"
    "* [Summarise documents](/Summarise) to extract key dates, "
    "people, actions and discussion for your principal \n"
    "* [Chat with documents](/Chat) to answer questions about your box's content \n"
)
