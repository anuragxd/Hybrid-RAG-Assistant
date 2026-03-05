from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def run_ragas_evaluation(questions: list, answers: list, contexts: list):
    """
    Evaluates the RAG pipeline.
    contexts should be a list of lists of strings (the retrieved text).
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    dataset = Dataset.from_dict(data)
    
    # Needs OPENAI_API_KEY by default, or configured to use Langchain wrapped Groq
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
    )
    return result