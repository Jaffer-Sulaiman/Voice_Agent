import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


#general imports
import os
from dotenv import load_dotenv

# import streamlit
import streamlit as st

#langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from ingest_database import vectorstore
from langchain_core.tools import tool
from langchain.agents import AgentExecutor , create_tool_calling_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader




#initiate embeddings model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
# Using the recommended embedding model for Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# --- LangChain RAG Chain ---
# System prompt to define the agent's persona
SYSTEM_PROMPT = """You are a friendly and professional voice agent for Apex Builders, a construction company.
Answer questions concisely and in a helpful tone based *only* on the provided context.
If the context does not contain the answer, politely state that you don't have enough information.
If a user asks to speak to a person or schedule a call, you should acknowledge their request and state that you will schedule a callback and notify the responsible person.
Do not make up information.

Context:
{context}
"""

# Define the prompt for the RAG chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


PERSIST_DIRECTORY = "./chroma_db"
# Only load the existing database, never re-ingest!
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=2)
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
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🦜")
st.title("🦜 Agentic RAG Chatbot")

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
user_question = st.chat_input("How are you?")

# did the user submit a prompt?
if user_question:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))


    # invoking the agent
    result = agent_executor.invoke({"input": user_question, "chat_history":st.session_state.messages})

    ai_message = result["output"]

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))