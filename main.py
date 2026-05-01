from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.retrievers.bm25 import BM25Retriever

from src.data_loader import load_developer_docs, get_qdrant_vector_store

load_dotenv()


def ask(query_engine, question: str):
    """Helper function to display question, answer and sources nicely."""
    print(f"\n{'='*85}")
    print(f"Q: {question}")
    print(f"{'='*85}")
    
    response = query_engine.query(question)
    
    print(f"\nA: {response}\n")
    print(f"Sources ({len(response.source_nodes)} chunks used):")
    
    for i, node in enumerate(response.source_nodes, 1):
        source = node.metadata.get("source", "unknown")
        url = node.metadata.get("url", "unknown")
        score = getattr(node, 'score', "N/A")
        print(f"  {i}. [{source}] {url} (score: {score})")
    
    return response


if __name__ == "__main__":
    print("\n=== ContextIQ RAG Pipeline - Day 9 ===\n")

    # 1. Load documents
    print("Loading documents...")
    documents = load_developer_docs(chunk_size=1024, chunk_overlap=200)

    # 2. Initialize Qdrant
    vector_store = get_qdrant_vector_store()

    # 3. Parse documents into nodes (required for BM25)
    print("\nSplitting documents into nodes...")
    splitter = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"  Parsed {len(nodes)} nodes from {len(documents)} documents")

    # 4. Create Vector Index with Qdrant
    print("\nCreating VectorStoreIndex with Qdrant...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context
    )
    print("  [OK] Vector index created")

    # 5. Setup Hybrid Search (Vector + BM25)
    print("\nSetting up Hybrid Search (Vector + BM25)...")
    
    vector_retriever = index.as_retriever(similarity_top_k=8)
    
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, 
        similarity_top_k=6
    )

    # Combine both retrievers
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        mode="reciprocal_rerank",
        num_queries=1,
        use_async=False,
    )

    # Create Query Engine
    query_engine = RetrieverQueryEngine.from_args(fusion_retriever)
    print("  [OK] Hybrid Query Engine ready\n")

    # 6. Run test queries
    print("--- Running Test Queries ---")
    test_questions = [
        "How do I use dependency injection in FastAPI?",
        "Explain how useState works in React with example",
        "How do I create a simple AWS Lambda function in Python?",
        "How can I secure my FastAPI app with JWT?",
        "What is the difference between useEffect and useReducer?"
    ]

    for question in test_questions:
        ask(query_engine, question)
