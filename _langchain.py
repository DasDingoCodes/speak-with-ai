
# Here we are importing the necessary classes, functions from langchain library.
import openai
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import TextGen
from langchain.memory import ConversationBufferMemory
import os

from dotenv import load_dotenv

load_dotenv()


# Loaded via load_dotenv()
# Here we are setting up the OpenAI API key.
# os.environ['OPENAI_API_KEY'] = 'api-key'


# Here we are creating a prompt template for the conversation. 
# The prompt template is a list of messages that will be used to generate the conversation.
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


# Here we are creating a LLM (OpenAI in our case).
# llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))
llm = TextGen("localhost:7860/")
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)


# Here we are creating a function to generate a podcast content.
# The function takes three parameters: `prompt`, `podcaster` and `guest`.
# The `prompt` parameter is the topic of the podcast. Ex. Elon Musk joins Lex Fridman in conversation about AI, Autopilot, Neuralink, Tesla, and his personal history.
# The `podcaster` parameter is the name of the podcaster. Ex. Lex Fridman.
# The `guest` parameter is the name of the guest. Ex. Elon Musk.
def get_response(prompt, podcaster, guest):
    _prompt = f"""

    Generate a podcast between {podcaster} and {guest}. They are discussing about {prompt}.
    
    """

    response = conversation.predict(input=_prompt)
    
    return response
