import os
from dotenv import load_dotenv
from uuid import uuid4

#from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch


# Load environment variables from .env file
load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Embeddings Initialization ---
# create embedding function
embeddings_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True} # Normalizing all embeddings in range of 0-1

)

# --- ChromaDB Setup ---
# Load the knowledge base document
KNOWLEDGE_BASE_PATH = "documents/Knowlede_Base_Apex_Builders_50.md"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

vector_store = Chroma(
    collection_name="va_collection",
    embedding_function=embeddings_function,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
)

def ingest_knowledge_base():
    print("Starting ingestion process...")
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            knowledge_base_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Knowledge base file not found at: {KNOWLEDGE_BASE_PATH}. Please ensure it's in the 'docs' folder.")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=knowledge_base_content)]
    splits = text_splitter.split_documents(docs)

    print("&&&&&&&&&&&&&&")
    print(splits)

    uuids = [str(uuid4()) for _ in range(len(splits))]
    print("Uids: ", uuids)

    # Create a ChromaDB vector store from the document chunks
    print(f"Ingesting {len(splits)} document chunks into ChromaDB...")
    try:
        vector_store.add_documents(documents=splits, ids=uuids)
        print(f"Ingestion complete. Vector store saved to '{CHROMA_PERSIST_DIRECTORY}'.")

    except Exception as e:

        print(f"\n---!!! A CRITICAL ERROR OCCURRED DURING INGESTION !!!---")
        print(f"Error: {e}")

if __name__ == "__main__":
    ingest_knowledge_base()

