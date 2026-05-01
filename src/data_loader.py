import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from qdrant_client import QdrantClient

load_dotenv()

# === SETUP ===

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

Settings.embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
)


def configure_chunking(chunk_size: int = 1024, chunk_overlap: int = 200):
    """
    chunk_size: how many tokens per chunk (bigger = more context, but less precise retrieval)
    chunk_overlap: tokens shared between consecutive chunks (prevents losing context at boundaries)

    Think of it like reading a book with sticky notes:
      - chunk_size = how much text fits on each sticky note
      - chunk_overlap = how many lines you copy to the NEXT sticky note so you don't lose context
    """
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    print(f"  Chunking config: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

FASTAPI_URLS = [
    "https://fastapi.tiangolo.com/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/",
    "https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/",
    "https://fastapi.tiangolo.com/advanced/dependencies/",
    "https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/",
    "https://fastapi.tiangolo.com/advanced/security/oauth2-scopes/",
    "https://fastapi.tiangolo.com/tutorial/path-params/",
    "https://fastapi.tiangolo.com/tutorial/query-params/",
]

REACT_URLS = [
    "https://react.dev/",
    "https://react.dev/reference/react/useState",
    "https://react.dev/reference/react/useEffect",
    "https://react.dev/learn/managing-state",
    "https://react.dev/reference/react/useContext",
    "https://react.dev/reference/react/useReducer",
    "https://react.dev/learn/render-and-commit",
    "https://react.dev/learn/keeping-components-pure",
]

AWS_URLS = [
    "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/getting-started.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/python-context.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/services-s3.html",   # new
]

def load_developer_docs(chunk_size: int = 1024, chunk_overlap: int = 200):
    """Load docs from multiple domains and tag each document with its source."""
    configure_chunking(chunk_size, chunk_overlap)

    reader = SimpleWebPageReader(html_to_text=True)

    all_docs = []
    sources = {
        "FastAPI": FASTAPI_URLS,
        "React": REACT_URLS,
        "AWS": AWS_URLS
        }

    for source_name, urls in sources.items():
        print(f"  Loading {source_name} docs ({len(urls)} pages)...")
        docs = reader.load_data(urls)
        for doc in docs:
            doc.metadata["source"] = source_name
        all_docs.extend(docs)
        print(f"    [OK] Loaded {len(docs)} documents from {source_name}")

    print(f"\n  Total: {len(all_docs)} documents loaded")
    for doc in all_docs:
        url = doc.metadata.get("url", "unknown")
        preview = doc.text[:80].replace("\n", " ").strip()
        preview = preview.encode("ascii", errors="replace").decode("ascii")
        print(f"    [{doc.metadata['source']}] {url}")
        print(f"      Preview: {preview}...")

    return all_docs


def get_qdrant_vector_store():
    # Increase timeout: collection creation on first run (especially on
    # Windows + Docker Desktop) can exceed the default 5s httpx timeout.
    client = QdrantClient(host="localhost", port=6333, timeout=60)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="contextiq_docs" #collection name in qdrant
    )
    print("[OK] Connected to Qdrant Docker (persistent storage)")
    return vector_store
    
    