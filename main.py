from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex

from src.data_loader import load_fastapi_docs

load_dotenv()


if __name__ == "__main__":
    print("Starting ContextIQ RAG...")

    documents = load_fastapi_docs()

    print("creating vector store index...")
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    response = query_engine.query("How to use FastAPI dependency injection with classes?")

    print("\n" + "="*60)
    print("Answer:")
    print(response)
    print("="*60 + "\n")
    print(f"\nSources Used: {len(response.source_nodes)}")



