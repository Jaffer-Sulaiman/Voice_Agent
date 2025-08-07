import os
from dotenv import load_dotenv

import nest_asyncio
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import SupabaseVectorStore

# import supabase db
from supabase.client import Client, create_client

# Apply the asyncio patch
nest_asyncio.apply()

# Load environment variables from a .env file (if it exists)
load_dotenv()

# initiating supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Streamlit App Setup ---
st.set_page_config(page_title="Apex Builders RAG Test", layout="centered")
st.title("💬 Apex Builders RAG Test Agent")
st.caption("🚀 RAG-powered conversational agent using Streamlit, LangChain, and ChromaDB.")

# Check for API key and provide a warning if not set
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in a .env file.")
else:
    # --- RAG Chain Initialization with Cache ---
    @st.cache_resource
    def initialize_rag_chain():
        """Initializes and caches the RAG chain to avoid re-computation."""
        try:
            # --- LLM and Embeddings Initialization ---
            # Explicitly pass the api_key to prevent a fallback to Application Default Credentials
            llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            
            # initiating vector store
            vector_store = SupabaseVectorStore(
                embedding=embeddings,
                client=supabase,
                table_name="documents",
                query_name="match_documents",
            )
            retriever = vector_store.as_retriever()

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
                ("placeholder", "{agent_scratchpad}"),
            ])
            
            document_chain = create_stuff_documents_chain(llm, qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            return retrieval_chain
        
        except Exception as e:
            st.error(f"An error occurred during RAG chain initialization: {e}")
            return None

    # Initialize the RAG chain and check if it was successful
    rag_chain = initialize_rag_chain()
    
    if rag_chain:
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from session state
        for message in st.session_state.messages:
            with st.chat_message(message["sender"]):
                st.markdown(message["text"])

        # Chat input and logic
        if prompt := st.chat_input("Ask about Apex Builders..."):
            st.session_state.messages.append({"sender": "user", "text": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Agent is typing..."):
                    # Convert history for LangChain format
                    chat_history_lc = []
                    for msg in st.session_state.messages:
                        if msg['sender'] == 'user':
                            chat_history_lc.append(HumanMessage(content=msg['text']))
                        else: # Assuming 'assistant' sender
                            chat_history_lc.append(AIMessage(content=msg['text']))
                    
                    # Invoke the RAG chain
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": chat_history_lc
                    })
                    
                    agent_response = response["answer"]
                    st.markdown(agent_response)
                    st.session_state.messages.append({"sender": "assistant", "text": agent_response})