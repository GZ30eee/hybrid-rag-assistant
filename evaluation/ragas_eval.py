import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import context_relevancy, answer_relevancy, faithfulness, context_recall
from core.hybrid_retriever import retrieve_documents
from core.llm_interface import generate_answer

def load_test_data(file_path: str = "evaluation/test_data/sample_qa.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def run_ragas_evaluation():
    """Run RAGAS evaluation on the RAG pipeline."""
    # Ensure we have some documents indexed (you can pre-index a test corpus)
    # For this example, we assume the session state has a corpus.
    # In CI, you might mock this.
    
    # Load test data
    test_df = load_test_data()
    
    # Prepare columns required by RAGAS
    questions = []
    answers = []
    contexts = []
    
    for _, row in test_df.iterrows():
        query = row['question']
        # Retrieve contexts
        doc_results = retrieve_documents(query, k=3)
        context_texts = [res.text for res in doc_results]
        # Generate answer
        context = "\n\n".join(context_texts)
        # Use a dummy chat history
        llm_response = generate_answer(query, context, "Bulleted List", doc_results, chat_history=[])
        answer = llm_response.get("short", "")
        
        questions.append(query)
        answers.append(answer)
        contexts.append(context_texts)
    
    # Build dataset for RAGAS
    dataset = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })
    
    # Evaluate
    result = evaluate(
        dataset=dataset,
        metrics=[context_relevancy, answer_relevancy, faithfulness, context_recall]
    )
    
    print("Evaluation Results:")
    print(result)
    
    # Save results
    result.to_csv("evaluation/results.csv")
    return result

if __name__ == "__main__":
    run_ragas_evaluation()
