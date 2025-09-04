import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("openai_api_key")
openai.api_base = os.getenv("openai_api_base")
openai.api_type = os.getenv("openai_api_type")
openai.api_version = os.getenv("openai_api_version")


# Folder containing PDFs
pdf_folder = "data"  # Change to your folder name

# Check number of PDF files in the folder
pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files in '{pdf_folder}' folder.")
if not pdf_files:
    print("No PDF files found. Exiting.")
    exit(0)

# Load and split documents
documents = []
for filename in pdf_files:
    loader = PyPDFLoader(os.path.join(pdf_folder, filename))
    try:
        docs = loader.load()
        documents.extend(docs)
        print(f"Loaded: {filename}")
    except Exception as e:
        print(f"Skipped {filename}: {e}")

if not documents:
    print("No text extracted from any PDF. Exiting.")
    exit(0)



splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"Total documents loaded: {len(documents)}")
if not chunks:
    print("No text chunks created from documents. Exiting.")
    exit(0)

import chromadb
from chromadb.utils import embedding_functions

# Create embeddings and store in ChromaDB

client = chromadb.PersistentClient(path="tjx_vectorstore")
embeddings = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-ada-002",
    deployment_id="gpt-4o",
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
    # chunk_size=1
)

# vectorstore = Chroma.from_documents(embeddings, persist_directory="tjx_vectorstore")

vectorstore = Chroma(persist_directory="tjx_vectorstore", embedding_function=embeddings)

vectorstore.persist()
print("Vector store created and persisted.")