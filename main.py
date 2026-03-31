import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings


load_dotenv()


# === LLM SETUP ===
llm = Groq(
    model="llama-3-70b-8192t",
    api_key=os.getenv("GROQ_API_KEY"),
)

# === Embedding Model ===

embed_model = FastEmbedEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
)

Settings.llm = llm
Settings.embed_model = embed_model


print("ContextIQ setup successful!")
print(f"Using LLM: {llm.model}")
print("Embedding model loaded - ready for RAG!")





