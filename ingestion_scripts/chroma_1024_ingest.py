# import basics
from uuid import uuid4
import torch

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# initiate embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)


# load pdf docs from folder 'documents'
loader = PyPDFDirectoryLoader("documents")

# split the documents in multiple chunks
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# initiating vector store
CHROMA_PERSIST_DIRECTORY = "./chroma_db_bge_1024"

vector_store = Chroma(
    collection_name="va_collection",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
)

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