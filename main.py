# filename: main.py
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# FastAPI setup
app = FastAPI()

# --- Environment Variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")


# --- LLM and Embeddings Initialization ---
# Using Gemini 2.5 Flash for the LLM
# Explicitly pass the api_key here
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=GEMINI_API_KEY)
# Using the recommended embedding model for Gemini
# Explicitly pass the api_key here as well
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

# --- ChromaDB Setup and RAG ---
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
    raise FileNotFoundError(f"ChromaDB directory not found at: {CHROMA_PERSIST_DIRECTORY}. Please run ingest_data.py first to create the database.")

# Load the existing ChromaDB vector store
vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

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

# Create a document chain to combine retrieved documents with the prompt
document_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create the retrieval chain
# This chain takes user input, retrieves relevant documents, and then passes them to the LLM
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- FastAPI Endpoints ---

# Pydantic model for the request body
class ChatRequest(BaseModel):
    user_prompt: str
    history: List[Dict[str, str]] # [{'sender': 'user', 'text': '...'}]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles chat requests, performs RAG, calls the Gemini LLM, and returns the response.
    """
    try:
        # Convert history for LangChain format
        chat_history_lc = []
        for msg in request.history:
            if msg['sender'] == 'user':
                chat_history_lc.append(HumanMessage(content=msg['text']))
            else: # Assuming 'agent' sender
                chat_history_lc.append(AIMessage(content=msg['text']))

        # Invoke the RAG chain
        # The 'input' here is the current user's prompt
        # The 'chat_history' is the accumulated conversation
        response = retrieval_chain.invoke({
            "input": request.user_prompt,
            "chat_history": chat_history_lc
        })

        # The response from retrieval_chain contains 'answer' and 'context' (retrieved docs)
        agent_response = response["answer"]
        return {"response": agent_response}

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

# Mount the static directory to serve the HTML file
# The HTML file will be served at the root URL '/'
app.mount("/", StaticFiles(directory=".", html=True), name="static")

