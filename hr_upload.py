import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import pinecone
import uuid
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# Configure API keys and initialize clients
genai.configure(api_key="AIzaSyBVPq6QUM156sNEXpPDJaPycmUMdOHZfOo")
pc = Pinecone(
    api_key="pcsk_5Fk6cs_L8N9FbdByyJqFFHsr3J6K1RLBonzMav3jmzGY9TA18zave7MWkDDtnSWnp3Vj6r",
    environment="us-west1-gcp"
)

index_name = "hr-policy-index"

# Initialize Pinecone index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(index_name)

# Initialize Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs efficiently
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    return "\n".join([page.get_text() for page in doc])

# Function to chunk the text for Pinecone insertion
def chunk_text(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to generate embeddings
def generate_embeddings(text):
    return model.encode(text, convert_to_tensor=False).tolist()

# Insert PDF data into Pinecone (batch process for performance)
def insert_to_pinecone(pdf_text, filename):
    chunks = chunk_text(pdf_text)
    batch = [(str(uuid.uuid4()), generate_embeddings(chunk), {'filename': filename, 'text': chunk}) for chunk in chunks]
    index.upsert(batch)  # Insert in bulk for efficiency

# Function to handle file upload
def handle_file_upload():
    uploaded_file = st.file_uploader("Upload HR Policy PDF", type=["pdf"])
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        insert_to_pinecone(text, uploaded_file.name)
        st.success(f"File '{uploaded_file.name}' uploaded and processed successfully.")
    else:
        st.info("Please upload an HR policy PDF file.")

# Streamlit app UI
def app():
    st.title("HR Policy Document Upload")
    handle_file_upload()

# Run the app
if __name__ == "__main__":
    app()