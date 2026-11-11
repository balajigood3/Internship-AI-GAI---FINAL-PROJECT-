import os
from dotenv import load_dotenv
from grounded_assistant.core import (
    GroundedAssistant, OpenAIGenerator, SentenceTransformerEmbedder,
    VectorRetriever, load_text_documents,
)
from grounded_assistant.evaluation import run_evaluation

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env to run the live evaluation.")
    chunks = load_text_documents("data")
    retriever = VectorRetriever(SentenceTransformerEmbedder())
    retriever.index(chunks)
    assistant = GroundedAssistant(retriever, OpenAIGenerator())
    report = run_evaluation(assistant, "eval/questions.json", "eval/results.json")
    print(f"Pass rate: {report['pass_rate']:.0%}")
    print(f"Tokens: {report['total_input_tokens']} in / {report['total_output_tokens']} out")
    print(f"Estimated cost: ${report['estimated_total_cost_usd']:.6f}")

if __name__ == "__main__":
    main()
