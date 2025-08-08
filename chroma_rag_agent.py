# import basics
import os
from dotenv import load_dotenv
import nest_asyncio

# Apply the asyncio patch
nest_asyncio.apply()

# import streamlit
import streamlit as st

# import langchain
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch


# load environment variables
load_dotenv()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# initiating embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    #model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)


# initiating vector store
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
#CHROMA_PERSIST_DIRECTORY = "./chroma_db_bge_1024"

vector_store = Chroma(
    collection_name="va_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
)
 
# initiating llm
llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# System prompt to define the agent's persona
SYSTEM_PROMPT = """You are a friendly and professional voice agent for Apex Builders, a construction company.
Answer questions concisely and in a helpful tone based *only* on the information you retrieve using your tools.
If the the information you retrieve using your tools does not contain the answer, politely state that you don't have enough information.
If a user asks to speak to a person or schedule a call, you should acknowledge their request and state that you will schedule a callback and notify the responsible person.
Do not make up information.
"""

# Define the prompt for the RAG chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

no_of_retrival_docs = 10
#no_of_retrival_docs = 5
# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=no_of_retrival_docs)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# initiating the agent
agent = create_tool_calling_agent(llm, tools, qa_prompt)

# create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# initiating streamlit app
st.set_page_config(page_title="Apex Builders RAG Test", layout="centered")
st.title("Apex Builders RAG Test Agent")
st.caption("🚀 RAG-powered conversational agent using Streamlit, LangChain, and ChromaDB.")


# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# create the bar where we can type messages
user_question = st.chat_input("Ask about your requirements")


# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        with st.spinner("Agent is typing..."):

            # invoking the agent
            result = agent_executor.invoke({
                "input": user_question,
                "chat_history": st.session_state.messages
            })

            ai_message = result["output"]
            st.markdown(ai_message)
            
    st.session_state.messages.append(AIMessage(ai_message))

    
