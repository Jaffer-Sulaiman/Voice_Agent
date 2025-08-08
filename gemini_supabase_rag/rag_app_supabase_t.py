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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool

# import supabase db
from supabase.client import Client, create_client

# load environment variables
load_dotenv()

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiating embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# initiating vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
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


# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
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
user_question = st.chat_input("How are you?")


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

    
