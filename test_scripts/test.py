import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

#imports for LangChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Define the model name and keyword arguments
model_name = "sentence-transformers/all-MiniLM-L6-v2" 
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True} # Normalizing all embeddings in range of 0-1

CHROMA_PERSIST_DIRECTORY = "./chroma_langchain_db"

 # create embedding function
embeddings_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=embeddings_function,
    persist_directory=CHROMA_PERSIST_DIRECTORY,
)


document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

#embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#vectors = embeddings_model.embed_documents([doc.page_content for doc in documents])

print(f"Ingesting {len(documents)} document chunks into ChromaDB...")
vector_store.add_documents(documents=documents, ids=uuids)
#vector_store. # Ensure data is saved to disk
#vector_store.add_documents(documents=documents, embeddings=vectors, ids=uuids)
print(f"Ingestion complete. Vector store saved to '{CHROMA_PERSIST_DIRECTORY}'.")