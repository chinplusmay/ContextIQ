import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.readers.web import SimpleWebPageReader

load_dotenv()

# === SETUP ===

Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

Settings.embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
)


# === load fastapi docs ===

def load_fastapi_docs():
    print("Loading FastAPI docs...")

    urls = [
        "https://fastapi.tiangolo.com/",
        "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "https://fastapi.tiangolo.com/tutorial/dependencies/classes-as-dependencies/",
        "https://fastapi.tiangolo.com/advanced/dependencies/",
    ]

    reader = SimpleWebPageReader(html_to_text=True)
    documents = reader.load_data(urls)

    print(f"loaded {len(documents)} docs from fast api docs")
    for i, doc in enumerate(documents[:3]): # preview of 3 only
        print(f"Document{i+1}: {doc.metadata.get('url', 'No URL')} - {len(doc.text)} characters")
        
    return documents