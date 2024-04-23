import json
import uuid
from datetime import date, datetime

import streamlit as st
from streamlit_feedback import streamlit_feedback

from redbox.llm.prompts.core import CORE_REDBOX_PROMPT
from redbox.models.chat import ChatMessage
from streamlit_app.utils import (
    init_session_state,
    replace_doc_ref,
    submit_feedback,
)

st.set_page_config(page_title="Redbox Copilot - Ask the Box", page_icon="📮", layout="wide")

ENV = init_session_state()
USER = st.session_state.backend.get_user()
INITIAL_CHAT_PROMPT = [
    ChatMessage(
        text=CORE_REDBOX_PROMPT.format(
            current_date=date.today().isoformat(),
            user_info=USER.summary(),
        ),
        role="system",
    ),
    ChatMessage(text="Hi, I'm Redbox Copilot. How can I help you?", role="ai"),
]

# Model selector


def change_selected_model():
    st.session_state.backend.set_llm(
        model=st.session_state.model_select,
        max_tokens=st.session_state.model_params["max_tokens"],
        temperature=st.session_state.model_params["temperature"],
    )
    st.toast(f"Loaded {st.session_state.model_select}")


model_select = st.sidebar.selectbox(
    "Select Model",
    options=st.session_state.available_models,
    on_change=change_selected_model,
    key="model_select",
)

feedback_kwargs = {
    "feedback_type": "thumbs",
    "optional_text_label": "What did you think of this response?",
    "on_submit": submit_feedback,
}

clear_chat = st.sidebar.button("Clear Chat")

if "chat_id" not in st.session_state or clear_chat:
    st.session_state.backend.set_chat_prompt(init_messages=INITIAL_CHAT_PROMPT)
    st.session_state.chat_id = uuid.uuid4()
    # clear feedback
    for key in list(st.session_state.keys()):
        if str(key).startswith("feedback_"):
            del st.session_state[key]

if "ai_message_markdown_lookup" not in st.session_state:
    st.session_state["ai_message_markdown_lookup"] = {}


def render_citation_response(response):
    cited_chunks = [
        (
            chunk.metadata["parent_doc_uuid"],
            chunk.metadata["url"],
            (chunk.metadata["page_numbers"] if "page_numbers" in chunk.metadata else None),
        )
        for chunk in response["input_documents"]
    ]
    cited_chunks = set(cited_chunks)
    cited_files = st.session_state.backend.get_files([uuid.UUID(x[0]) for x in cited_chunks])
    page_numbers = [x[2] for x in cited_chunks]

    for j, page_number in enumerate(page_numbers):
        if isinstance(page_number, str):
            page_numbers[j] = json.loads(page_number)

    response_markdown = replace_doc_ref(
        str(response["output_text"]),
        cited_files,
        page_numbers=page_numbers,
        flexible=True,
    )

    return response_markdown


now_formatted = datetime.now().isoformat().replace(".", "_")

# st.sidebar.download_button(
#     label="Download Conversation",
#     data=json.dumps(
#         [x["text"] for x in st.session_state.messages],
#         indent=4,
#         ensure_ascii=False,
#     ),
#     file_name=(f"redboxai_conversation_{st.session_state.user_uuid}" f"_{now_formatted}.json"),
# )

# message_count = len(st.session_state.messages)

# for i, msg in enumerate(st.session_state.messages):
#     if msg["role"] == "system":
#         continue
#     avatar_map = {"human": "🧑‍💻", "ai": "📮", "user": "🧑‍💻", "assistant": "📮"}
#     if hash(msg["text"]) in st.session_state.ai_message_markdown_lookup:
#         with st.chat_message(msg["role"], avatar=avatar_map[msg["role"]]):
#             st.markdown(
#                 st.session_state.ai_message_markdown_lookup[hash(msg["text"])],
#                 unsafe_allow_html=True,
#             )
#     else:
#         st.chat_message(msg["role"], avatar=avatar_map[msg["role"]]).write(msg["text"])

#     if st.session_state.messages[i]["role"] in ["ai", "assistant"] and i > 1:
#         streamlit_feedback(
#             **feedback_kwargs,
#             key=f"feedback_{i}",
#             kwargs={
#                 "input": [msg["text"] for msg in st.session_state.messages],
#                 "chain": st.session_state.messages[0:i],
#                 "output": st.session_state.messages[i]["text"],
#                 "creator_user_uuid": USER.uuid,
#             },
#         )

avatar_map = {"human": "🧑‍💻", "ai": "📮", "user": "🧑‍💻", "assistant": "📮", "AIMessageChunk": "📮"}

for chat_message in st.session_state.backend.get_chat(chat_uuid=st.session_state.chat_id).messages:
    st.chat_message(chat_message.type, avatar=avatar_map[chat_message.type]).write(chat_message.content)

if prompt := st.chat_input():
    st.chat_message("user", avatar=avatar_map["user"]).write(prompt)
    with st.chat_message("assistant", avatar=avatar_map["assistant"]):
        reply_stream = st.session_state.backend.simple_schat(input=prompt, chat_uuid=st.session_state.chat_id)
        st.write_stream(reply_stream)


# if prompt := st.chat_input():
#     st.session_state.messages.append(ChatMessage(text=prompt, role="human"))
#     st.chat_message("user", avatar=avatar_map["user"]).write(prompt)

#     with st.chat_message("assistant", avatar=avatar_map["assistant"]):
#         response_stream_text = st.empty()

#         response = st.session_state.backend.rag_chat(
#             chat_history=st.session_state.messages,
#         )

#         response_final_markdown = render_citation_response(response)

#         response_stream_text.empty()
#         response_stream_text.markdown(response_final_markdown, unsafe_allow_html=True)

#     st.session_state.messages.append(ChatMessage(text=response, role="ai"))

#     streamlit_feedback(
#         **feedback_kwargs,
#         key=f"feedback_{len(st.session_state.messages) - 1}",
#         kwargs={
#             "input": [msg["text"] for msg in st.session_state.messages],
#             "chain": st.session_state.messages[0:-1],
#             "output": st.session_state.messages[-1]["text"],
#             "creator_user_uuid": USER.uuid,
#         },
#     )

#     # Store the markdown response for later rendering
#     # Done to avoid needing file references from llm_handler
#     st.session_state.ai_message_markdown_lookup[hash(response["output_text"])] = response_final_markdown
