import sys
sys.path.insert(0, '/test_scripts')

from test import vector_store

results = vector_store.similarity_search(
    "The stock market is down 500 points",
    k=2,
    filter={"source": "news"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")