from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from src.data_loader import load_developer_docs, inspect_naive_chunks

load_dotenv()


def ask(query_engine, question: str):
    """Ask a question and display the answer with source attribution."""
    print(f"\n{'='*80}")
    print(f"Q: {question}")
    print(f"{'='*80}")

    response = query_engine.query(question)
    print(f"\nA: {response}\n")

    print(f"Sources ({len(response.source_nodes)} chunks used):")
    for i, node in enumerate(response.source_nodes, 1):
        source = node.metadata.get("source", "unknown")
        url = node.metadata.get("url", "unknown")
        score = node.score if node.score else "N/A"
        print(f"  {i}. [{source}] {url} (score: {score})")

    return response


if __name__ == "__main__":
    # Load documents from both domains
    print("Step 1: Loading documents...")
    documents = load_developer_docs(chunk_size=1024, chunk_overlap=200)

    # Inspect naive chunks
    print("\nStep 2: Inspecting naive chunks...")
    naive_chunks = inspect_naive_chunks(documents)

    # Build the vector index (this embeds all chunks)
    print("\nStep 3: Creating vector index (embedding all chunks)...")
    index = VectorStoreIndex.from_documents(documents)
    print("  [OK] Index created")

    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=3)

      # Test with questions from BOTH domains
    print("\n--- Testing cross-domain retrieval ---")

    ask(query_engine, "How do I use dependency injection in FastAPI?")
    ask(query_engine, "How does useState work in React?")
    ask(query_engine, "What is useEffect and when should I use it?")
    ask(query_engine, "What is a lambda function and how do I use it?")

