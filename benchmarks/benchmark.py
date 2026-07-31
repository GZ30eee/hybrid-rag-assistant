import time
import json
import statistics
from core.hybrid_retriever import retrieve_documents
from core.llm_interface import generate_answer

def benchmark_retrieval(queries: list, k: int = 5, iterations: int = 3):
    """Measure average retrieval latency."""
    times = []
    for query in queries:
        for _ in range(iterations):
            start = time.time()
            retrieve_documents(query, k)
            end = time.time()
            times.append(end - start)
    avg_time = statistics.mean(times)
    print(f"Average retrieval time for {len(queries)} queries (k={k}): {avg_time:.4f}s")
    return avg_time

def benchmark_llm(queries: list, context: str = "Sample context for testing.", iterations: int = 3):
    """Measure average LLM generation latency and token usage."""
    times = []
    token_usage = []
    for query in queries:
        for _ in range(iterations):
            start = time.time()
            response = generate_answer(query, context, "Bulleted List", [], chat_history=[])
            end = time.time()
            times.append(end - start)
            if response.get("token_usage"):
                token_usage.append(response["token_usage"]["total_tokens"])
    avg_time = statistics.mean(times)
    avg_tokens = statistics.mean(token_usage) if token_usage else 0
    print(f"Average LLM generation time: {avg_time:.4f}s, avg tokens: {avg_tokens:.2f}")
    return avg_time, avg_tokens

if __name__ == "__main__":
    # Example queries
    test_queries = ["What is RAG?", "How does hybrid search work?"]
    benchmark_retrieval(test_queries)
    benchmark_llm(test_queries)
