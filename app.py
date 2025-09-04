__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

st.set_page_config(layout='centered')
st.markdown("### Store Ops Assistant 🙋🏻‍♂️")
st.divider()

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("openai_api_key")
openai.api_base = os.getenv("openai_api_base")
openai.api_type = os.getenv("openai_api_type")
openai.api_version = os.getenv("openai_api_version")

# Convert to vectors
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

# Create embeddings and store in ChromaDB
client = chromadb.PersistentClient(path="tjx_vectorstore")
embeddings = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-ada-002",  # Use your actual embedding model name
    deployment_id="text-embedding-ada-002",  # Use your actual Azure embedding deployment name
    api_key=openai.api_key,
    api_base=openai.api_base,
    api_type=openai.api_type,
    api_version=openai.api_version,
)
# vectorstore = Chroma(persist_directory="tjx_vectorstore", embedding_function=embeddings)

# vectorstore.persist()
# print("Vector store created and persisted.")


collection = client.get_or_create_collection("langchain", embedding_function=embeddings)

# Check and print the number of documents in the collection
# try:
#     count_result = collection.count()
#     st.write(f"Number of documents in ChromaDB collection: {count_result}")
# except Exception as e:
#     st.write(f"Could not retrieve document count: {e}")


query = st.text_input("Ask a question:")
send = st.button("Send")

if query and send:
    # Query ChromaDB for relevant documents
    results = collection.query(query_texts=[query], n_results=2)
    # Extract text from results
    docs = []
    for doc_list in results.get("documents", []):
        docs.extend(doc_list)

    context = "\n\n".join(docs) if docs else "No relevant documents found."
    # Show the context being sent to the model
    # st.write("Context sent to model:")
    # st.write(context)

    # Use AzureOpenAI client for chat completions
    from openai import AzureOpenAI
    openai_client = AzureOpenAI(
        api_key=openai.api_key,
        api_version=openai.api_version,
        azure_endpoint=openai.api_base
    )

    # Compose prompt for Azure OpenAI
    messages = [
        {"role": "system", "content": "You are a Store Operations Assistant. Use the provided context to answer the user's question."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o",  # Or your preferred chat model deployment name
        messages=messages,
        temperature=0.5
    )
    st.write(response.choices[0].message.content)
