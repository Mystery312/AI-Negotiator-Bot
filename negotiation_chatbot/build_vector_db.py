# build_vector_db.py

import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import time

#  1. CONFIGURATION 
# Define the paths and model name.
# This makes it easy to change them later.
DATA_SOURCES_PATH = "./data_sources" # Folder where your raw PDF and TXT files are.
CHROMA_DB_PATH = "./chroma_db"     # Folder where the vector database will be stored.
COLLECTION_NAME = "negotiation_tactics"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # A good, fast starting model.

#  2. HELPER FUNCTIONS 

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_text_from_txt(txt_path: str) -> str:
    """Extracts text from a TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {txt_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Cleans the extracted text by removing artifacts."""
    # Remove excessive newlines and whitespace
    text = re.sub(r'\s*\n\s*', '\n', text).strip()
    # Remove hyphenation at the end of lines
    text = text.replace('-\n', '')
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

#  3. MAIN SCRIPT LOGIC 

def main():
    """
    Main function to process documents and build the vector database.
    """
    print(" Starting Vector Database Build ")
    start_time = time.time()

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # Load the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Embedding model loaded.")

    # Initialize the vector database client
    print(f"Initializing vector database at: {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("Vector database initialized.")

    # Get a list of all files to process
    all_files = [f for f in os.listdir(DATA_SOURCES_PATH) if os.path.isfile(os.path.join(DATA_SOURCES_PATH, f))]
    
    total_chunks_indexed = 0

    # Process each file
    for filename in all_files:
        file_path = os.path.join(DATA_SOURCES_PATH, filename)
        print(f"\nProcessing file: {filename}...")

        # Extract text based on file type
        raw_text = ""
        if filename.lower().endswith('.pdf'):
            raw_text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.txt'):
            raw_text = extract_text_from_txt(file_path)
        else:
            print(f"Skipping unsupported file type: {filename}")
            continue

        if not raw_text:
            print(f"No text extracted from {filename}. Skipping.")
            continue
        
        # Clean and chunk the text
        cleaned_text = clean_text(raw_text)
        chunks = text_splitter.split_text(cleaned_text)

        if not chunks:
            print(f"No chunks created for {filename}. Skipping.")
            continue

        print(f"Split into {len(chunks)} chunks. Generating embeddings...")

        # Generate embeddings for all chunks in the current document
        embeddings = embedding_model.encode(chunks, show_progress_bar=True).tolist()

        # Create unique IDs and metadata for each chunk
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in range(len(chunks))]
        
        # Add the data to the collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        total_chunks_indexed += len(chunks)
        print(f"Successfully indexed {len(chunks)} chunks for {filename}.")

    end_time = time.time()
    print("\n Vector Database Build Complete ")
    print(f"Total files processed: {len(all_files)}")
    print(f"Total chunks indexed: {total_chunks_indexed}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Database saved in '{CHROMA_DB_PATH}' directory.")


if __name__ == "__main__":
    # Ensure the data sources directory exists
    if not os.path.exists(DATA_SOURCES_PATH):
        os.makedirs(DATA_SOURCES_PATH)
        print(f"Created directory '{DATA_SOURCES_PATH}'. Please add your PDF and TXT files there.")
    else:
        main()