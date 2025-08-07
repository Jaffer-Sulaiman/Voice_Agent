import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
import chromadb


CHROMA_PERSIST_DIRECTORY = "./chroma_langchain_db"
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)

 # create embedding function
embedding_function = GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_API_KEY
    )

# data to an existing collection.
collection = client.get_or_create_collection(
        name="test_collection", embedding_function=embedding_function
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
    document_1.page_content,
    document_2.page_content,
    document_3.page_content,
    document_4.page_content,
    document_5.page_content,
    document_6.page_content,
    document_7.page_content,
    document_8.page_content,
    document_9.page_content,
    document_10.page_content,
]
uuids = [str(uuid4()) for _ in range(len(documents))]

#embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#vectors = embeddings_model.embed_documents([doc.page_content for doc in documents])

print(f"Ingesting {len(documents)} document chunks into ChromaDB...")
collection.add(ids= uuids, documents=documents)
#vector_store.add_documents(documents=documents, embeddings=vectors, ids=uuids)
print(f"Ingestion complete. Vector store saved to '{CHROMA_PERSIST_DIRECTORY}'.")