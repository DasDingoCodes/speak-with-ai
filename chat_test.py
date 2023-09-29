from langchain.chains import LLMChain
from langchain.llms import OpenAI, TextGen
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from elevenlabs import Voice, generate, save, set_api_key
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
import base64


load_dotenv()
set_api_key(os.environ["ELEVEN_API_KEY"])


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true" style="width: 100%;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="StreamlitChatMessageHistory", page_icon="📖")
st.title("📖 StreamlitChatMessageHistory")

"""
A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation.
The messages are stored in Session State across re-runs automatically. You can view the contents of Session State
in the expander below. View the
[source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py).
"""

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

if "voice" not in st.session_state:
    voice = Voice(
        voice_id="VqDG5DtZTPpVg2XUtO3u"
    )
    st.session_state.voice = voice

view_messages = st.expander("View the message contents in session state")

llm = TextGen(model_url="http://localhost:5000")

# # Get an OpenAI API Key before continuing
# if "openai_api_key" in st.secrets:
#     openai_api_key = st.secrets.openai_api_key
# else:
#     openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Enter an OpenAI API Key to continue")
#     st.stop()
# llm = OpenAI(openai_api_key=openai_api_key)

# Set up the LLMChain, passing in memory
template = """You are an AI chatbot having a conversation with a human.

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Render current messages from StreamlitChatMessageHistory
for i, msg in enumerate(msgs.messages):
    with st.chat_message(msg.type):
        st.write(msg.content)
        path_audio_file = Path(f"{i}.mp3")
        if path_audio_file.is_file():
            st.audio(str(path_audio_file))

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    response = llm_chain.run(prompt)
    audio = generate(text=response, voice=st.session_state.voice)
    msg_index = len(msgs.messages) -1
    path_audio_file = f"{msg_index}.mp3"
    save(audio, path_audio_file)
    with st.chat_message("ai"):
        st.write(response)
        autoplay_audio(str(path_audio_file))

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)