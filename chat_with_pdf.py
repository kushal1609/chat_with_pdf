import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY", "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")

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("Hugging Face API token is missing. Add it to your .env file or replace 'your_huggingface_api_key' with your key.")

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store."""
    if not text_chunks:
        raise ValueError("No text chunks available for vectorization.")
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store():
    """Load FAISS vector store."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

def get_conversational_chain():
    """Initialize Hugging Face question-answering pipeline."""
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
        use_auth_token=HUGGINGFACE_API_TOKEN,
    )
    return qa_pipeline

def answer_question(user_question):
    """Retrieve context and generate answer for user question."""
    try:
        vector_store = load_vector_store()
        docs = vector_store.similarity_search(user_question, k=3)  # Top 3 matches
        if not docs:
            st.warning("No relevant context found for your question. Please try rephrasing.")
            return
        
        # Combine content for context
        context = " ".join([doc.page_content for doc in docs])

        # Display relevant sections
        st.write("### Relevant PDF Sections")
        for doc in docs:
            st.markdown(f"**Context:** {doc.page_content}")

        # Generate response using Hugging Face QA pipeline
        qa_pipeline = get_conversational_chain()
        response = qa_pipeline({"question": user_question, "context": context})
        st.success("Answer: " + response["answer"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    """Streamlit main function."""
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.title("Chat with PDF using Hugging Face 🤗")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.header("Upload and Process PDFs")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return
            
            try:
                with st.spinner("Processing..."):
                    text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed successfully. You can now ask questions.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

    # Question input
    user_question = st.text_input("Ask a question about the content of the PDFs")
    if user_question:
        answer_question(user_question)

if __name__ == "__main__":
    main()
