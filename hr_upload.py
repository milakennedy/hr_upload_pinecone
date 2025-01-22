import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import uuid
from pinecone import Pinecone, ServerlessSpec

# Configure Pinecone API keys and initialize
pinecone = Pinecone(
    api_key="pcsk_5Fk6cs_L8N9FbdByyJqFFHsr3J6K1RLBonzMav3jmzGY9TA18zave7MWkDDtnSWnp3Vj6r",
    environment="us-west1-gcp"  # Use your Pinecone environment
)

index_name = "document-index"

# Ensure the index exists
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Ensure it matches the embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust cloud/region if needed
    )
index = pinecone.Index(index_name)

# Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        raise ValueError(f"Error reading PDF file: {e}")

# Function to chunk text
def chunk_text(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to generate embeddings
def generate_embeddings(text):
    return model.encode(text, convert_to_tensor=False).tolist()

# Insert PDF data into Pinecone
def insert_to_pinecone(pdf_text, filename):
    chunks = chunk_text(pdf_text)
    batch = [
        (str(uuid.uuid4()), generate_embeddings(chunk), {'filename': filename, 'text': chunk})
        for chunk in chunks
    ]
    index.upsert(batch)

# Handle file uploads
def handle_file_upload():
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"Processing file: {uploaded_file.name}")
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                insert_to_pinecone(pdf_text, uploaded_file.name)
                st.success(f"File '{uploaded_file.name}' processed and indexed successfully.")
            except Exception as e:
                st.error(f"Error processing '{uploaded_file.name}': {e}")
    else:
        st.info("Please upload one or more PDF files.")

# Streamlit app
def app():
    st.title("Document Upload System")
    st.header("Upload PDF Documents")
    handle_file_upload()

# Run the app
if __name__ == "__main__":
    app()
