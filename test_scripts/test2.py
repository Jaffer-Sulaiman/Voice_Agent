from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

vector_store = Chroma(
    collection_name="test_collection",
    embedding_function=FakeEmbeddings(size=10),
    persist_directory="./test_chroma_db",
)

doc = Document(page_content="test", metadata={"source": "test"})
vector_store.add_documents([doc])
print("Success!")