# Chat with PDF using Hugging Face 🤖

# Object
This project enables users to upload PDF files and ask context-specific questions about their content. It leverages Hugging Face models for embeddings and question answering, making it an interactive and intelligent document assistant.
# Features
PDF Upload and Text Extraction: Upload multiple PDF files, extract their text, and process it into manageable chunks.
Semantic Search: Uses vector embeddings to find the most relevant sections of the PDF for a given question.
Question Answering: Provides detailed answers to user queries using Hugging Face's deepset/roberta-base-squad2 model.
Streamlit UI: A user-friendly interface for uploading PDFs, viewing content, and interacting with the system.
# Installation
Prerequisites : (1) Python 3.8 or higher
                (2) A Hugging Face API token (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)
# Project Structure
app.py --> Main application script ,
requirements.txt --> Python dependencies ,
README.md --> Project documentation ,               
faiss_index --> Folder to store the FAISS vector store(generated during runtime)
# Technologies Used
Streamlit: For building the user interface.
Hugging Face Transformers: For question answering (deepset/roberta-base-squad2).
LangChain: For text embedding and chunking.
FAISS: For semantic search and vector database.
PyPDF2: For extracting text from PDF files.
Python-dotenv: For managing environment variables.
# How It Works
Upload PDFs: Users upload PDF files, which are processed to extract text.
Text Splitting: The extracted text is divided into smaller chunks for better processing.
Vectorization: Each text chunk is embedded into a vector space using sentence-transformers/all-MiniLM-L6-v2.
Semantic Search: When a question is asked, the system retrieves the most relevant chunks from the vector store.
Answer Generation: The Hugging Face QA pipeline generates a response based on the retrieved context.

# Demo
![image](https://github.com/user-attachments/assets/3b3b757a-4f9b-4eae-8470-1daee9a90cd8)

![image](https://github.com/user-attachments/assets/54deb600-70b3-4728-87a6-35ca28195b1f)
