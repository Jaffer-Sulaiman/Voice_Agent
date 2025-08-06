import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


#general imports
import os
from dotenv import load_dotenv

#langchain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#load env
load_dotenv()
# --- Environment Variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

#load pdf docs from the folder
loader = PyPDFDirectoryLoader("documents")

#split the doc into multiple chunks
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

#initiate embeddings model
# Using the recommended embedding model for Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#init database and store your embeddings
# for this we created a local ChromaDB instance in a directory named "chroma_db"
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
#retriever = vectorstore.as_retriever()



