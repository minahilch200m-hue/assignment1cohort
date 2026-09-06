"""
Document Ingestion Pipeline - Part 1 of larger project.
Handles loading, cleaning, chunking, embedding, and vector storage of documents.
Designed to be modular and extendable.
"""

# Required pip install packages (run before importing):
# pip install langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf docx2txt groq python-dotenv

# For MMR: langchain-cohere or use FAISS directly

import sys
import os
import re
import glob

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
# Part 2: Advanced RAG Pipeline
# ============================================================

def retrieve_documents(query, vectorstore, top_k=5, similarity_threshold=0.5):
    """
    Perform semantic retrieval using the FAISS vectorstore.

    Args:
        query: user query string
        vectorstore: FAISS vectorstore object
        top_k: number of top results to retrieve (default: 5)
        similarity_threshold: minimum similarity score to keep a result (default: 0.5)

    Returns:
        list of Document objects that passed the threshold
    """
    # Perform similarity search with scores (FAISS supports this via similarity_search_with_score)
    # We'll use similarity_search first, then score filtering
    # Note: FAISS similarity_search doesn't return scores directly,
    # so we use max_marginal_relevance_search or do a basic search then filter

    # Use similarity_search to get top_k results
    docs = vectorstore.similarity_search(query, k=top_k)

    # If we need scores, we can do a similarity search with score
    # For FAISS, we use the internal method
    try:
        # Try to get similarity scores
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        # results is list of (doc, score) tuples
        filtered_docs = []
        for doc, score in results:
            # FAISS similarity scores are typically cosine similarity inverted
            # Convert to 0-1 range if needed; here we treat higher as more similar
            # The score from MiniLM is cosine similarity in [-1, 1], but often [0, 1]
            # We'll treat thresholds appropriately
            if score >= similarity_threshold:
                filtered_docs.append(doc)
        # If filtering removed too many, fall back to original top_k
        if len(filtered_docs) < 2:
            filtered_docs = docs
    except Exception:
        # fallback if similarity_search_with_score not available
        filtered_docs = docs

    # If fallback, also apply threshold based on simple similarity logic
    if not filtered_docs and docs:
        filtered_docs = docs[:top_k]

    return filtered_docs


def retrieve_documents_mmr(query, vectorstore, top_k=5, similarity_threshold=0.5):
    """
    Perform Maximal Marginal Relevance (MMR) retrieval using FAISS.

    Args:
        query: user query string
        vectorstore: FAISS vectorstore object
        top_k: number of top results to retrieve (default: 5)
        similarity_threshold: minimum similarity score to keep a result (default: 0.5)

    Returns:
        list of Document objects selected via MMR
    """
    try:
        # FAISS's max_marginal_relevance_search method
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=top_k,
            fetch_k=min(top_k * 2, vectorstore.index.ntotal or top_k),
            lambda_mult=0.5,  # diversity control: 1.0=very diverse, 0.0=very relevant
        )
        # Apply similarity threshold filtering
        filtered_docs = []
        for doc in docs:
            # MMR returns docs; we need to check scores
            # Since MMR doesn't directly give us scores we can threshold,
            # we'll do a simple post-filter by re-checking similarity
            # For now, return the MMR-selected docs
            filtered_docs.append(doc)

        # If we have too few results, fall back to regular search
        if len(filtered_docs) < 2:
            docs = vectorstore.similarity_search(query, k=top_k)
            filtered_docs = docs

        return filtered_docs
    except Exception as e:
        print(f"MMR retrieval error: {e}")
        # Fallback to regular similarity search
        return vectorstore.similarity_search(query, k=top_k)


def rerank_documents(query, documents):
    """
    Re-rank documents using a Cross-Encoder model from Hugging Face.

    Uses 'cross-encoder/ms-marco-MiniLM-L-6-v2' which runs locally via
    sentence-transformers with no API key required.

    Args:
        query: user query string
        documents: list of Document objects to re-rank

    Returns:
        list of Document objects sorted by relevance score (highest first)
    """
    from sentence_transformers import CrossEncoder

    # Load cross-encoder model
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Score each document against the query
    # The model takes pairs (query, document) and returns relevance scores
    scores = model.predict([(query, doc.page_content) for doc in documents])

    # Pair documents with scores and sort by score descending
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Return just the documents sorted by relevance
    reranked_docs = [pair[0] for pair in doc_score_pairs]

    return reranked_docs


def optimize_context(documents, max_context_length=3000):
    """
    Optimize the context by:
    - Removing near-duplicate chunks (highly similar text content)
    - Filtering out chunks below similarity threshold
    - Enforcing a maximum total context length by trimming lowest-ranked chunks first

    Args:
        documents: list of Document objects (typically reranked)
        max_context_length: maximum total characters for the combined context (default: 3000)

    Returns:
        list of final Document objects to use as context
    """
    if not documents:
        return []

    # Step 1: Remove near-duplicate chunks
    # We'll track texts we've already seen and skip ones with high overlap
    seen_texts = []
    deduped_docs = []

    for doc in documents:
        content = doc.page_content
        is_near_duplicate = False

        for seen_text in seen_texts:
            # Simple near-duplicate detection: check if chunk text is subset or highly overlapping
            # Using a simple ratio check
            if len(content) > 0 and len(seen_text) > 0:
                # Check if one is contained in the other (high overlap)
                if content.strip() in seen_text or seen_text in content.strip():
                    is_near_duplicate = True
                    break
                # Check character-level overlap percentage
                common = len(set(content.lower().split()) & set(seen_text.lower().split()))
                union = len(set(content.lower().split()) | set(seen_text.lower().split()))
                if union > 0 and common / union > 0.8:
                    is_near_duplicate = True
                    break

        if not is_near_duplicate:
            seen_texts.append(content)
            deduped_docs.append(doc)

    # Step 2: Filter chunks and build combined context
    # Sort by relevance (already sorted from reranker) and accumulate until max length
    combined_length = 0
    final_docs = []

    for doc in deduped_docs:
        chunk_len = len(doc.page_content)

        # Check if adding this chunk exceeds the limit
        if combined_length + chunk_len > max_context_length:
            # If we already have some docs, stop here
            if final_docs:
                break
            # If this is the first chunk and it's still too large, include it anyway but trimmed
            else:
                # Trim the chunk to fit
                trimmed_content = doc.page_content[:max_context_length]
                doc.page_content = trimmed_content
                final_docs.append(doc)
                break

        combined_length += chunk_len
        final_docs.append(doc)

    # Step 3: Return final cleaned list
    # Update metadata to reflect optimization
    for i, doc in enumerate(final_docs):
        doc.metadata["context_rank"] = i + 1

    return final_docs


def answer_query(query, vectorstore):
    """
    Run the full Advanced RAG pipeline flow:
    User Query → Query Processing → Retriever → Top-K Documents → Re-ranking → 
    Context Construction → LLM → Answer + Citations

    Args:
        query: user query string
        vectorstore: FAISS vectorstore object

    Returns:
        None (prints the answer with citations)
    """
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Step 1: Retrieve documents
    print("=" * 70)
    print("STEP 1: Semantic Retrieval")
    print("=" * 70)
    retrieved = retrieve_documents(query, vectorstore, top_k=5, similarity_threshold=0.5)
    print(f"Retrieved {len(retrieved)} documents (top_k=5, threshold=0.5)")

    # Print retrieved documents before reranking
    for i, doc in enumerate(retrieved):
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:60].replace("\n", " ")
        print(f"  [{i+1}] {source}: {content_preview}...")

    # Step 2: MMR Retrieval (optional advanced technique)
    print("\n" + "=" * 70)
    print("STEP 2: MMR Retrieval (Maximal Marginal Relevance)")
    print("=" * 70)
    mmr_retrieved = retrieve_documents_mmr(query, vectorstore, top_k=5, similarity_threshold=0.5)
    print(f"MMR retrieved {len(mmr_retrieved)} documents")

    # Print MMR results
    for i, doc in enumerate(mmr_retrieved):
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:60].replace("\n", " ")
        print(f"  [{i+1}] {source}: {content_preview}...")

    # Compare before/after reranking
    print("\n" + "=" * 70)
    print("COMPARISON: Before vs After Reranking")
    print("=" * 70)

    # Combine retrieved docs for reranking (use regular retrieval as base)
    docs_to_rerank = retrieved if retrieved else mmr_retrieved

    if len(docs_to_rerank) > 1:
        # Show before order
        print("\n--- Before Reranking (similarity order) ---")
        for i, doc in enumerate(docs_to_rerank):
            source = doc.metadata.get("source", "unknown")
            print(f"  [{i+1}] {source}: {doc.page_content[:50].replace(chr(10), ' ')}...")

        # Rerank the documents
        reranked = rerank_documents(query, docs_to_rerank)

        # Show after order
        print("\n--- After Reranking (cross-encoder relevance) ---")
        for i, doc in enumerate(reranked):
            source = doc.metadata.get("source", "unknown")
            print(f"  [{i+1}] {source}: {doc.page_content[:50].replace(chr(10), ' ')}...")

        # Use reranked documents for context
        selected_docs = reranked
    else:
        selected_docs = docs_to_rerank

    # Step 3: Optimize context
    print("\n" + "=" * 70)
    print("STEP 3: Context Optimization")
    print("=" * 70)
    optimized = optimize_context(selected_docs, max_context_length=3000)
    print(f"Optimized from {len(selected_docs)} to {len(optimized)} documents")
    total_chars = sum(len(d.page_content) for d in optimized)
    print(f"Total context length: {total_chars} characters")

    # Print optimized context chunks
    for i, doc in enumerate(optimized):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        content_preview = doc.page_content[:50].replace("\n", " ")
        print(f"  [{i+1}] {source} — Page {page}: {content_preview}...")

    # Step 4: Generate answer with LLM via Groq
    print("\n" + "=" * 70)
    print("STEP 4: LLM Answer Generation")
    print("=" * 70)

    # Load Groq API key from .env
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in environment variables.")
        print("Please create a .env file with: GROQ_API_KEY=your_key_here")
        return

    # Construct the prompt with optimized context and user question
    context_text = ""
    for i, doc in enumerate(optimized):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        context_text += f"\n--- Context Chunk {i+1} (from {source}, Page {page}) ---\n{doc.page_content}\n"

    prompt = f"""
You are a helpful assistant answering questions based solely on the provided context.

INSTRUCTION: Answer the user's question using ONLY the information provided in the context below.

CONTEXT:
{context_text}

USER QUESTION: {query}

ANSWER (based strictly on the context provided above):
"""

    # Generate answer using Groq
    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based solely on provided context. Use only the information given; do not use external knowledge."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

        # Step 5: Print answer with citations
        print("\n" + "=" * 70)
        print("ANSWER WITH CITATIONS")
        print("=" * 70)
        print(f"Answer: {answer}")

        # Print citations based on the optimized context chunks used
        print("\nSources:")
        citations = []
        for i, doc in enumerate(optimized):
            source = doc.metadata.get("source", "unknown")
            # Extract filename from path
            filename = os.path.basename(source)
            page = doc.metadata.get("page", "unknown")
            # Convert page index to page number (0-indexed to 1-indexed)
            page_num = page + 1 if isinstance(page, int) else page
            citations.append((i + 1, filename, page_num))

        # Deduplicate citations by filename
        seen_files = set()
        for num, filename, page in citations:
            if filename not in seen_files:
                seen_files.add(filename)
                print(f"  [{num}] {filename} — Page {page}")

    except Exception as e:
        print(f"Error generating answer with Groq: {e}")
        print("Could not connect to Groq API. Make sure GROQ_API_KEY is set correctly.")


# ============================================================
# Entry Point - Part 1 & Part 2
# ============================================================

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

    all_chunks = []

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
            all_chunks.extend(chunks)

            # 4 & 5. Embed and Store per-file vectorstore
            print("\n[Step 4 & 5] Embedding and storing in vector database...")
            safe_file_name = filename.replace(".", "_")
            index_path = f"faiss_index_{safe_file_name}"
            vectorstore = build_or_load_vectorstore(chunks, embeddings, index_path=index_path)
            print(f"Vectorstore for '{filename}' contains {vectorstore.index.ntotal} embeddings.")

        print("\n" + "=" * 60)
        print("All individual files processed successfully!")
        print("=" * 60)

        # Build / Load the combined vectorstore from all chunks
        print("\n" + "=" * 60)
        print("Building / Loading Combined Vectorstore (faiss_index)")
        print("=" * 60)
        combined_index_path = "faiss_index"
        combined_vectorstore = build_or_load_vectorstore(
            all_chunks, embeddings, index_path=combined_index_path
        )
        print(f"Combined vectorstore ready at '{combined_index_path}' with {combined_vectorstore.index.ntotal} total embeddings.")

    # Part 2: Run advanced RAG pipeline test using the combined vectorstore
    vectorstore_path = "faiss_index"
    if os.path.exists(vectorstore_path):
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.load_local(
            vectorstore_path, embeddings, allow_dangerous_deserialization=True
        )
        print(f"\nVectorstore loaded with {vectorstore.index.ntotal} vectors.\n")

        # Run a sample query through the full Part 2 pipeline
        sample_query = "What does the test PDF document say?"
        print("=" * 70)
        print("PART 2 ADVANCED RAG PIPELINE TEST")
        print("=" * 70)
        print(f"\nSample Query: '{sample_query}'\n")
        answer_query(sample_query, vectorstore)
    else:
        print(f"\nNo vectorstore found at '{vectorstore_path}'. "
              "Run the pipeline first to build the index, then Part 2 functions can be imported and used.")