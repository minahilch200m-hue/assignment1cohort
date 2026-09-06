"""
Document Ingestion Pipeline - Part 1 of larger project.
Handles loading, cleaning, chunking, embedding, and vector storage of documents.
Designed to be modular and extendable.
"""

# Required pip install packages (run before importing):
# pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf docx2txt

import os
import re
import glob

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# 1. Document Loading
# ============================================================

def load_document(file_path):
    """
    Load a document from file path using the appropriate LangChain loader
    based on file extension.

    Supported extensions: .pdf, .txt, .docx

    Returns a list of LangChain Document objects.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .txt, .docx")

    docs = loader.load()
    return docs


def load_all_documents_from_folder(folder_path):
    """
    Load all supported documents from a folder recursively.
    Returns a single list of Document objects.
    """
    all_docs = []
    supported_extensions = [".pdf", ".txt", ".docx"]

    # Find all files recursively
    for ext in supported_extensions:
        pattern = os.path.join(folder_path, f"**/*{ext}")
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            print(f"Loading: {file_path}")
            docs = load_document(file_path)
            all_docs.extend(docs)

    return all_docs


# ============================================================
# 2. Document Cleaning
# ============================================================

def clean_documents(docs):
    """
    Clean loaded documents by:
    - Removing empty or near-empty pages
    - Collapsing excessive whitespace/newlines into single spaces
    - Removing duplicate page/content blocks
    - Fixing broken words from line-break hyphenation (e.g. "docu-\nment" -> "document")
    - Normalizing metadata keys to lowercase and consistent format

    Returns a cleaned list of Document objects.
    """
    cleaned = []

    for doc in docs:
        content = doc.page_content

        # Fix broken words from line-break hyphenation
        # Pattern: word-\nword -> combinedword
        content = re.sub(r"(\w+)-\n(\w+)", r"\1\2", content)

        # Collapse excessive whitespace and newlines into single spaces
        content = re.sub(r"\s+", " ", content).strip()

        # Remove empty or near-empty pages (very short content)
        if len(content) < 10:
            continue  # skip very short content

        # Remove duplicate content blocks (keep first occurrence only)
        # We'll track seen contents to de-duplicate
        is_duplicate = False
        for existing in cleaned:
            if existing.page_content == content:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        # Normalize metadata keys to lowercase and consistent format
        new_metadata = {}
        for key, value in doc.metadata.items():
            new_key = key.lower().strip().replace(" ", "_")
            new_metadata[new_key] = value

        # Create new cleaned document
        from langchain_core.documents import Document as LCDoc
        cleaned_doc = LCDoc(page_content=content, metadata=new_metadata)
        cleaned.append(cleaned_doc)

    return cleaned


# ============================================================
# 3. Chunking
# ============================================================

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    """
    Split documents into chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Args:
        docs: list of Document objects
        chunk_size: size of each chunk in characters (default: 500)
        chunk_overlap: overlap between chunks (default: 50)

    Returns:
        list of chunk Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(docs)
    return chunks


# ============================================================
# 4. Embeddings
# ============================================================

def get_embedding_model():
    """
    Return a Hugging Face local embedding model.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' which runs entirely locally
    with no API key required.

    Returns:
        HuggingFaceEmbeddings model object
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    # device='cpu' ensures it runs on CPU without GPU requirements
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )
    return embeddings


# ============================================================
# 5. Vector Database (FAISS)
# ============================================================

def build_or_load_vectorstore(chunks, embeddings, index_path="faiss_index"):
    """
    Build a new FAISS vectorstore from chunks and embeddings,
    or load an existing saved index.

    If a saved FAISS index already exists at index_path, it loads it
    instead of rebuilding. This ensures persistence between application
    restarts.

    Args:
        chunks: list of Document chunks
        embeddings: HuggingFaceEmbeddings model object
        index_path: path to save/load the FAISS index (default: "faiss_index")

    Returns:
        FAISS vectorstore object
    """
    # Check if saved index exists
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from: {index_path}")
        # Load the existing vectorstore
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    # Build new FAISS index from chunks and embeddings
    print("Building new FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the index to disk for persistence
    print(f"Saving FAISS index to: {index_path}")
    vectorstore.save_local(index_path)

    return vectorstore


# ============================================================
# Main Pipeline Script
# ============================================================

def run_pipeline(folder_path, index_path="faiss_index"):
    """
    Run the full document ingestion pipeline:
    load -> clean -> chunk -> embed -> store

    Args:
        folder_path: path to folder containing documents to process
        index_path: path for FAISS index persistence

    Returns:
        FAISS vectorstore with all processed documents
    """
    print("=" * 60)
    print("Document Ingestion Pipeline - Part 1")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[Step 1] Loading documents...")
    docs = load_all_documents_from_folder(folder_path)
    print(f"Loaded {len(docs)} document pages/sections.")

    # Step 2: Clean documents
    print("\n[Step 2] Cleaning documents...")
    cleaned_docs = clean_documents(docs)
    print(f"After cleaning: {len(cleaned_docs)} documents remain.")

    # Step 3: Chunk documents
    print("\n[Step 3] Chunking documents...")
    chunks = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)
    print(f"Created {len(chunks)} chunks.")

    # Step 4: Get embeddings model
    print("\n[Step 4] Loading embedding model...")
    embeddings = get_embedding_model()
    print("Embedding model loaded successfully.")

    # Step 5: Build/load vectorstore
    print("\n[Step 5] Building/loading vectorstore...")
    vectorstore = build_or_load_vectorstore(chunks, embeddings, index_path=index_path)
    print(f"Vectorstore ready with {vectorstore.index.ntotal} total vectors.")

    # Step 6: Sample similarity search test
    print("\n[Step 6] Running sample similarity search test...")
    if chunks:
        # Use the first chunk's content as a query test
        test_query = chunks[0].page_content[:50]
        print(f"Query: '{test_query}...'")
        results = vectorstore.similarity_search(test_query, k=1)
        if results:
            print(f"Top result: {results[0].page_content[:80]}...")
            print("✅ Pipeline end-to-end test PASSED")
        else:
            print("⚠️  No results returned from similarity search")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return vectorstore


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # Directory containing sample documents
    sample_folder = os.path.join(os.path.dirname(__file__), "sample_docs")
    os.makedirs(sample_folder, exist_ok=True)

    # Create a sample text file if no docs are found
    if not any(f.endswith((".pdf", ".txt", ".docx")) for f in os.listdir(sample_folder)):
        sample_file = os.path.join(sample_folder, "sample.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(
                "This is a sample document for the pipeline test. "
                "It contains multiple sentences that will be loaded, "
                "cleaned, chunked, embedded, and stored in a FAISS vectorstore. "
                "The pipeline should handle PDF, TXT, and DOCX files automatically "
                "based on file extension. This is just a test to verify the "
                "end-to-end functionality works correctly."
            )

    print("=" * 60)
    print("Running Document Ingestion Pipeline on sample_docs")
    print("=" * 60)

    # Load embedding model once for processing
    embeddings = get_embedding_model()

    # Get all supported files from sample_docs folder
    supported_extensions = [".pdf", ".txt", ".docx"]
    all_files = [
        f for f in os.listdir(sample_folder)
        if os.path.isfile(os.path.join(sample_folder, f)) and os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not all_files:
        print(f"No supported files found in '{sample_folder}'.")
    else:
        for filename in sorted(all_files):
            file_path = os.path.join(sample_folder, filename)
            file_ext = os.path.splitext(filename)[1].lower()

            print("\n" + "-" * 60)
            print(f"Processing File: {filename}")
            print(f"Detected File Type: {file_ext}")
            print("-" * 60)

            # 1. Load document
            print(f"\n[Step 1] Loading document ({file_ext})...")
            docs = load_document(file_path)
            print(f"Loaded {len(docs)} document page(s)/section(s).")

            # 2. Clean document
            print("\n[Step 2] Cleaning document...")
            cleaned_docs = clean_documents(docs)
            print(f"After cleaning: {len(cleaned_docs)} document(s) remain.")

            # 3. Chunk document
            print("\n[Step 3] Chunking document...")
            chunks = chunk_documents(cleaned_docs, chunk_size=500, chunk_overlap=50)
            print(f"Created {len(chunks)} chunks for '{filename}'.")

            # 4 & 5. Embed and Store
            print("\n[Step 4 & 5] Embedding and storing in vector database...")
            safe_file_name = filename.replace(".", "_")
            index_path = f"faiss_index_{safe_file_name}"
            vectorstore = build_or_load_vectorstore(chunks, embeddings, index_path=index_path)
            print(f"Vectorstore for '{filename}' contains {vectorstore.index.ntotal} embeddings.")

    print("\n" + "=" * 60)
    print("All files processed successfully!")
    print("=" * 60)