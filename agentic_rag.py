"""
Part 3: Agentic RAG Workflow using LangGraph.
This module builds on Part 1 (pipeline.py - document ingestion) and Part 2 (RAG pipeline 
with retrieval, MMR, reranking, context optimization, and Groq citations).

Note: The full graph will be built iteratively. This file starts with the foundational pieces:
- GraphState TypedDict
- Query Analyzer node
- Retriever node
- Relevance Grader node
"""

import os
import sys
import json
import re
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
from groq import Groq

# Import retrieval functions from Part 2 pipeline
from pipeline import retrieve_documents, retrieve_documents_mmr, build_or_load_vectorstore, optimize_context

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in .env. Set it to use Groq LLM nodes.")
    groq_api_key_for_tests = None
else:
    groq_api_key_for_tests = GROQ_API_KEY


# ──────────────────────────────────────────────────────────────────────
# 1. GraphState — structured state passed between all nodes
# ──────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    """Structured state for the agentic RAG workflow."""
    question: str                          # Original user question
    rewritten_query: str                   # Query after rewriting (if any)
    documents: List[object]                # Retrieved document objects
    relevance_score: float                 # Score from relevance grader (0-1)
    relevance: bool                        # Whether docs are relevant to question
    answer: str                            # Generated answer from LLM
    citations: List[dict]                  # Citations for the answer
    retry_count: int                       # Number of query rewrites attempted


# ──────────────────────────────────────────────────────────────────────
# 2. Query Analyzer Node
# ──────────────────────────────────────────────────────────────────────
def query_analyzer_node(state: GraphState) -> GraphState:
    """
    Sends the user's question to the Groq LLM to determine:
    (a) whether the question requires document retrieval to answer, and
    (b) what type of information is being requested.
    
    Stores the analysis in the state.
    
    Args:
        state: Current GraphState with the user's question.
        
    Returns:
        Updated GraphState with analysis results.
    """
    question = state["question"]
    
    # Initialize fields that might not exist yet
    if "rewritten_query" not in state:
        state["rewritten_query"] = ""
    if "relevance_score" not in state:
        state["relevance_score"] = 0.0
    if "relevance" not in state:
        state["relevance"] = False
    if "retry_count" not in state:
        state["retry_count"] = 0
    
    if not groq_api_key_for_tests:
        # If no Groq key, default: assume retrieval is needed
        state["answer"] = "Groq API key not available; assuming retrieval is needed."
        state["citations"] = []
        return state
    
    try:
        client = Groq(api_key=groq_api_key_for_tests)
        
        prompt = f"""
You are analyzing a user query to determine if it requires document retrieval to answer, 
and what type of information is being requested.

User Question: "{question}"

Respond in JSON format with two fields:
1. "needs_retrieval": boolean (true if the question needs external documents to answer, false if it's a general knowledge question or can be answered without docs)
2. "info_type": string describing what type of information is requested (e.g., "factual", "summary", "comparison", "procedure", etc.)

Example output: {{"needs_retrieval": true, "info_type": "factual"}}
"""
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a query analyzer for a RAG system. Respond ONLY with valid JSON with 'needs_retrieval' (boolean) and 'info_type' (string) fields."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        
        # Parse the JSON response from the LLM
        response_content = response.choices[0].message.content
        # Try to find JSON object in the response
        analysis = None
        
        # Method 1: Try parsing the entire response as JSON
        try:
            analysis = json.loads(response_content.strip())
            if isinstance(analysis, dict) and "needs_retrieval" in analysis:
                pass  # success
            else:
                analysis = None
        except (json.JSONDecodeError, ValueError):
            analysis = None
        
        # Method 2: Extract JSON object if parsing failed
        if analysis is None:
            json_start = response_content.find("{")
            json_end = response_content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                try:
                    analysis = json.loads(json_str)
                    if not isinstance(analysis, dict):
                        analysis = None
                except (json.JSONDecodeError, ValueError):
                    analysis = None
        
        # Method 3: Regex-based extraction as last resort
        if analysis is None:
            import re
            match = re.search(r'\{"[^"]*"\s*:\s*"[^"]*"\s*,\s*"[^"]*"\s*:\s*"[^"]*"\}', response_content)
            if match:
                try:
                    analysis = json.loads(match.group(0))
                except (json.JSONDecodeError, ValueError):
                    analysis = None
        
        # Store analysis in state
        if analysis and isinstance(analysis, dict):
            state["relevance"] = analysis.get("needs_retrieval", True)
            state["relevance_score"] = float(analysis.get("needs_retrieval", True))  # Use 1.0 or 0.0 as proxy
            state["info_type"] = analysis.get("info_type", "unknown")
        else:
            # Fallback: default assumptions
            state["relevance"] = True
            state["relevance_score"] = 0.5
            state["info_type"] = "unknown"
        
    except Exception as e:
        print(f"Error in query_analyzer_node: {e}")
        # Default: assume retrieval is needed
        state["relevance"] = True
        state["relevance_score"] = 0.5
        state["info_type"] = "unknown"
    
    return state


# ──────────────────────────────────────────────────────────────────────
# 3. Retriever Node
# ──────────────────────────────────────────────────────────────────────
def retriever_node(state: GraphState) -> GraphState:
    """
    Node that retrieves documents using the pipeline's retrieval functions.
    Uses state["rewritten_query"] if it exists and is not empty,
    otherwise uses state["question"].

    Stores the retrieved documents in state["documents"].

    Args:
        state: Current GraphState.

    Returns:
        Updated GraphState with retrieved documents.
    """
    # Determine which query to use: rewritten_query if available and non-empty, else question
    query = state.get("rewritten_query", "") or state["question"]

    # Load vectorstore from FAISS index if not already in state
    vectorstore = state.get("_vectorstore", None)
    if vectorstore is None:
        from pipeline import get_embedding_model
        embeddings = get_embedding_model()
        vectorstore = build_or_load_vectorstore([], embeddings, index_path="faiss_index")
        state["_vectorstore"] = vectorstore

    # Use MMR retrieval for diverse results with the real vectorstore
    try:
        retrieved = retrieve_documents_mmr(query, vectorstore, top_k=5)
    except Exception as e:
        print(f"Retriever error: {e}")
        retrieved = []

    # Store retrieved documents in state
    state["documents"] = retrieved
    return state


# ──────────────────────────────────────────────────────────────────────
# 4. Relevance Grader Node
# ──────────────────────────────────────────────────────────────────────
def relevance_grader_node(state: GraphState) -> GraphState:
    """
    Node that sends state["documents"] and state["question"] to the Groq LLM
    to judge relevance. Asks the LLM to return a structured JSON response
    like {"relevant": true, "score": 0.91}.

    Parses the response and stores:
    - state["relevance_score"]: float score from the LLM (0-1)
    - state["relevance": bool] whether docs are relevant to the question

    Handles cases where the LLM response isn't perfectly formatted JSON
    gracefully (using try/except with a fallback default).

    Args:
        state: Current GraphState with documents and question.

    Returns:
        Updated GraphState with relevance score and decision.
    """
    if not groq_api_key_for_tests:
        # If no Groq key, default fallback
        state["relevance_score"] = 0.5
        state["relevance"] = False
        return state

    question = state["question"]
    documents = state.get("documents", [])

    if not documents:
        # No documents to grade
        state["relevance_score"] = 0.0
        state["relevance"] = False
        return state

    try:
        client = Groq(api_key=groq_api_key_for_tests)

        # Build a summary of documents for the prompt
        doc_summaries = []
        for i, doc in enumerate(documents[:5]):  # Limit to 5 docs for prompt size
            content = doc.page_content[:200].replace("\n", " ")
            doc_summaries.append(f"Document {i+1}: {content}...")

        doc_context = "\n".join(doc_summaries)

        prompt = f"""
You are a relevance grader for a RAG system. Given a user question and retrieved documents,
judge if the documents are relevant to the question. Respond with valid JSON having two fields:
1. "relevant": boolean (true if the documents contain information relevant to answering the question)
2. "score": float (0.0 to 1.0 confidence score)

User Question: "{question}"

Documents:
{doc_context}

Respond with JSON ONLY, e.g.: {{"relevant": true, "score": 0.91}}
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a relevance grader for a RAG system. Respond ONLY with valid JSON with 'relevant' (boolean) and 'score' (float 0-1) fields."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )

        response_content = response.choices[0].message.content

        # Parse the JSON response from the LLM
        parsed = None
        try:
            parsed = json.loads(response_content.strip())
            if isinstance(parsed, dict) and "relevant" in parsed and "score" in parsed:
                pass  # success
            else:
                parsed = None
        except (json.JSONDecodeError, ValueError):
            parsed = None

        # If first method failed, try extracting JSON from the response
        if parsed is None:
            json_start = response_content.find("{")
            json_end = response_content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                try:
                    parsed = json.loads(json_str)
                    if not isinstance(parsed, dict):
                        parsed = None
                except (json.JSONDecodeError, ValueError):
                    parsed = None

        # Fallback: regex-based extraction
        if parsed is None:
            import re
            match = re.search(r'\{[^}]*"relevant"[^}]*\}', response_content)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except (json.JSONDecodeError, ValueError):
                    parsed = None

        # Store results in state
        if parsed and isinstance(parsed, dict):
            state["relevance"] = bool(parsed.get("relevant", False))
            state["relevance_score"] = float(parsed.get("score", 0.5))
        else:
            # Fallback defaults
            state["relevance"] = False
            state["relevance_score"] = 0.5

    except Exception as e:
        print(f"Error in relevance_grader_node: {e}")
        # Graceful fallback
        state["relevance"] = False
        state["relevance_score"] = 0.5

    return state


# ──────────────────────────────────────────────────────────────────────
# 5. Query Rewriter Node
# ──────────────────────────────────────────────────────────────────────
def query_rewriter_node(state: GraphState) -> GraphState:
    """
    Node that, when relevance is False, sends the original question to the Groq LLM
    asking it to rewrite the query for better retrieval (different phrasing, more 
    specific terms related to the document domain).

    Stores the result in state["rewritten_query"] and increments state["retry_count"] by 1.

    Args:
        state: Current GraphState with question and relevance flag.

    Returns:
        Updated GraphState with rewritten query and incremented retry count.
    """
    if not groq_api_key_for_tests:
        # If no Groq key, just increment retry count and return
        state["retry_count"] += 1
        return state

    question = state["question"]
    
    try:
        client = Groq(api_key=groq_api_key_for_tests)
        
        prompt = f"""
You are a query rewriter for a RAG system. Given a user question that was deemed not 
relevant to the retrieved documents, rewrite the query using different phrasing and more 
specific terms related to the document domain to improve retrieval performance.

Original Question: "{question}"

Rewritten Question:
"""
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a query rewriter for a RAG system. Rewrite the user query to be more specific and effective for document retrieval. Respond with only the rewritten query text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        
        rewritten = response.choices[0].message.content.strip()
        state["rewritten_query"] = rewritten
        state["retry_count"] += 1
        
    except Exception as e:
        print(f"Error in query_rewriter_node: {e}")
        state["rewritten_query"] = question  # fallback: keep original
        state["retry_count"] += 1
    
    return state


# ──────────────────────────────────────────────────────────────────────
# 6. Generate Answer Node
# ──────────────────────────────────────────────────────────────────────
def generate_answer_node(state: GraphState) -> GraphState:
    """
    Node that generates an answer using the existing Part 2 context optimization and 
    Groq LLM answer generation functions from pipeline.py, producing an answer using 
    state["documents"].

    Stores the answer in state["answer"] and citations in state["citations"].

    Args:
        state: Current GraphState with documents and question.

    Returns:
        Updated GraphState with generated answer and citations.
    """
    if not groq_api_key_for_tests:
        state["answer"] = "Groq API key not available; cannot generate answer."
        state["citations"] = []
        return state

    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        state["answer"] = "No documents retrieved; cannot generate answer without context."
        state["citations"] = []
        return state

    try:
        from pipeline import rerank_documents, optimize_context
        
        # Step 1: Rerank documents for relevance
        reranked = rerank_documents(question, documents)
        
        # Step 2: Optimize context
        optimized = optimize_context(reranked, max_context_length=3000)
        
        # Step 3: Construct context text for Groq
        context_text = ""
        for i, doc in enumerate(optimized):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            context_text += f"\n--- Context Chunk {i+1} (from {source}, Page {page}) ---\n{doc.page_content}\n"
        
        # Step 4: Generate answer with Groq
        client = Groq(api_key=groq_api_key_for_tests)
        
        prompt = f"""
You are a helpful assistant answering questions based solely on the provided context.

INSTRUCTION: Answer the user's question using ONLY the information provided in the context below.
Do not use external knowledge.

CONTEXT:
{context_text}

USER QUESTION: {question}

ANSWER (based strictly on the context provided above):
"""
        
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
        
        # Step 5: Extract citations
        citations = []
        for i, doc in enumerate(optimized):
            source = doc.metadata.get("source", "unknown")
            filename = source.split("\\")[-1] if "\\" in source else source.split("/")[-1]
            page = doc.metadata.get("page", "unknown")
            page_num = page + 1 if isinstance(page, int) else page
            citations.append({"filename": filename, "page": page_num, "chunk": i + 1})
        
        # Deduplicate citations by filename
        seen_files = set()
        deduped_citations = []
        for c in citations:
            if c["filename"] not in seen_files:
                seen_files.add(c["filename"])
                deduped_citations.append(c)
        
        state["answer"] = answer
        state["citations"] = deduped_citations
        
    except Exception as e:
        print(f"Error in generate_answer_node: {e}")
        state["answer"] = f"Error generating answer: {e}"
        state["citations"] = []

    return state


# ──────────────────────────────────────────────────────────────────────
# 7. Citation Checker Node
# ──────────────────────────────────────────────────────────────────────
def citation_checker_node(state: GraphState) -> GraphState:
    """
    Node that sends the generated answer along with state["documents"] back to the Groq LLM,
    asking it to verify every claim in the answer is directly supported by the provided context.
    If unsupported claims are found, flags them by appending a note to the answer.

    Args:
        state: Current GraphState with answer and documents.

    Returns:
        Updated GraphState with potentially modified answer and citation info.
    """
    if not groq_api_key_for_tests:
        # No Groq key, default flag
        if state.get("answer") and "could not be verified" not in state["answer"].lower():
            state["answer"] += " [Note: some claims could not be verified against source documents]"
        return state

    question = state["question"]
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    
    if not answer:
        return state

    try:
        client = Groq(api_key=groq_api_key_for_tests)
        
        # Build document context for the checker
        doc_summaries = []
        for i, doc in enumerate(documents[:5]):
            content = doc.page_content[:300].replace("\n", " ")
            doc_summaries.append(f"Document {i+1}: {content}...")
        
        doc_context = "\n".join(doc_summaries)
        
        prompt = f"""
You are a citation verifier for a RAG system. Given a user question, a generated answer, 
and provided source documents, verify whether every claim in the answer is directly supported 
by the source documents. 

User Question: "{question}"
Generated Answer: "{answer}"

Documents:
{doc_context}

Respond with valid JSON having two fields:
1. "verified": boolean (true if all claims are directly supported by documents, false if some claims are unsupported)
2. "note": string (empty if verified, or a brief note explaining unsupported claims if not verified)

Respond with JSON ONLY, e.g.: {{"verified": true, "note": ""}} or {{"verified": false, "note": "Some claims about specific statistics could not be verified against the provided source documents."}}
"""
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a citation verifier for a RAG system. Respond ONLY with valid JSON with 'verified' (boolean) and 'note' (string) fields."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        
        response_content = response.choices[0].message.content
        
        # Parse the JSON response from the LLM
        parsed = None
        try:
            parsed = json.loads(response_content.strip())
            if isinstance(parsed, dict) and "verified" in parsed:
                pass  # success
            else:
                parsed = None
        except (json.JSONDecodeError, ValueError):
            parsed = None
        
        # Try extracting JSON from response
        if parsed is None:
            json_start = response_content.find("{")
            json_end = response_content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                try:
                    parsed = json.loads(json_str)
                    if not isinstance(parsed, dict):
                        parsed = None
                except (json.JSONDecodeError, ValueError):
                    parsed = None
        
        # Fallback: regex-based extraction
        if parsed is None:
            import re
            match = re.search(r'\{[^}]*"verified"[^}]*\}', response_content)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except (json.JSONDecodeError, ValueError):
                    parsed = None
        
        # Apply the result
        if parsed and isinstance(parsed, dict):
            if not parsed.get("verified", True):
                # Add note to answer about unverified claims
                note = parsed.get("note", " [Note: some claims could not be verified against source documents]")
                if note and note not in state["answer"]:
                    state["answer"] += note
        else:
            # Fallback: if we can't parse, add conservative note
            if "could not be verified" not in state["answer"].lower():
                state["answer"] += " [Note: some claims could not be verified against source documents]"
        
    except Exception as e:
        print(f"Error in citation_checker_node: {e}")
        if "could not be verified" not in state["answer"].lower():
            state["answer"] += " [Note: some claims could not be verified against source documents]"

    return state


# ──────────────────────────────────────────────────────────────────────
# 8. Build the LangGraph StateGraph
# ──────────────────────────────────────────────────────────────────────
def build_graph():
    """
    Build and return a compiled LangGraph StateGraph for the agentic RAG workflow.
    
    The workflow includes these nodes:
    - Query Analyzer
    - Retriever  
    - Relevance Grader
    - Query Rewriter
    - Generate Answer
    - Citation Checker
    
    Conditional routing:
    - After Relevance Grader: route to Generate Answer if relevance=True, else to Query Rewriter
    - After Query Rewriter: route back to Retriever
    - Max retry limit: if retry_count >= 2, force routing to Generate Answer
    
    Returns:
        Compiled LangGraph graph object.
    """
    from langgraph.graph import StateGraph, END
    
    # Define the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("query_analyzer", query_analyzer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("relevance_grader", relevance_grader_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("citation_checker", citation_checker_node)
    
    # Add edges (linear flow with conditionals)
    
    # 1. Query Analyzer -> Retriever
    workflow.add_edge("query_analyzer", "retriever")
    
    # 2. Retriever -> Relevance Grader
    workflow.add_edge("retriever", "relevance_grader")
    
    # 3. Relevance Grader -> Generate Answer (if relevance=True) or Query Rewriter (if relevance=False)
    def route_after_grader(state: GraphState) -> str:
        """Route after relevance grading: if relevant, generate answer; otherwise rewrite query."""
        if state.get("retry_count", 0) >= 2:
            # Max retry reached, force go to generate answer
            return "generate_answer"
        if state.get("relevance", False):
            return "generate_answer"
        else:
            return "query_rewriter"
    
    workflow.add_conditional_edges(
        "relevance_grader",
        route_after_grader,
        {
            "generate_answer": "generate_answer",
            "query_rewriter": "query_rewriter",
            END: END  # fallback
        }
    )
    
    # 4. Query Rewriter -> Retriever (loop back for another retrieval cycle)
    workflow.add_edge("query_rewriter", "retriever")
    
    # 5. Generate Answer -> Citation Checker
    workflow.add_edge("generate_answer", "citation_checker")
    
    # 6. Citation Checker -> End (or back to generate_answer if needed, but we'll end here)
    workflow.add_edge("citation_checker", END)
    
    # Set entry point
    workflow.set_entry_point("query_analyzer")
    
    # Compile the graph
    compiled_graph = workflow.compile()
    
    return compiled_graph


# ──────────────────────────────────────────────────────────────────────
# 9. Run Agentic RAG Function
# ──────────────────────────────────────────────────────────────────────
def run_agentic_rag(question: str):
    """
    Run the full agentic RAG workflow with the given question.
    Invokes the compiled graph and prints each node's execution.
    
    Args:
        question: User question string to process.
    """
    # Build the compiled graph
    graph = build_graph()
    
    # Initialize state
    initial_state: GraphState = {
        "question": question,
        "rewritten_query": "",
        "documents": [],
        "relevance_score": 0.0,
        "relevance": False,
        "answer": "",
        "citations": [],
        "retry_count": 0,
    }
    
    print(f"\n{'=' * 60}")
    print(f"Running: '{question}'")
    print(f"{'=' * 60}")
    
    # Run the graph
    final_state = graph.invoke(initial_state)
    
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")
    print(f"Final Answer: {final_state.get('answer', '')}")
    print(f"Citations: ", end="")
    if final_state.get("citations"):
        for c in final_state["citations"]:
            print(f"[{c.get('chunk', '')}] {c.get('filename', '')} — Page {c.get('page', '')}", end="; ")
        print()
    else:
        print("[]")
    print(f"Confidence: {final_state.get('relevance_score', 0.0):.2f}")
    print(f"Retries used: {final_state.get('retry_count', 0)}")
    print(f"{'=' * 60}\n")
    
    return final_state


# ──────────────────────────────────────────────────────────────────────
# 10. Final test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Part 3: Agentic RAG Full Workflow Test")
    print("=" * 60)
    
    # Test question a): Should pass relevance grading on first try
    print("\n" + "=" * 60)
    print("TEST a): 'What are the challenges of wind power?'")
    print("=" * 60)
    result_a = run_agentic_rag("What are the challenges of wind power?")
    
    # Test question b): Should fail relevance, trigger query rewriter loop, then generate answer
    print("\n" + "=" * 60)
    print("TEST b): 'What are the key benefits of electric vehicles?'")
    print("=" * 60)
    result_b = run_agentic_rag("What are the key benefits of electric vehicles?")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)