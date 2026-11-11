"""Run the fixed six-case evaluation without external API calls."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from grounded_assistant.core import (
    DocumentChunk, GroundedAssistant, VectorRetriever, NOT_FOUND
)
from grounded_assistant.evaluation import run_evaluation


class DemoEmbedder:
    TERMS = ["pump", "vibration", "compressor", "temperature", "pressure",
             "inspection", "maintenance", "safety", "filter", "bearing"]

    def encode(self, texts):
        return np.asarray([
            [float(term in text.lower()) for term in self.TERMS]
            for text in texts
        ], dtype=np.float32)


class DemoGenerator:
    """Deterministic stand-in used only to verify evaluation plumbing."""
    def generate(self, question, passages):
        q = question.lower()
        citation = f"[{passages[0].source}, page {passages[0].page}]"
        if "normal" in q and "temperature" in q:
            return NOT_FOUND, 30, 12
        if "pump" in q or "vibration" in q:
            return f"Record vibration readings and operating pressure {citation}.", 75, 18
        if "compressor" in q or "110" in q:
            return f"The sample report records 110 degrees Celsius {citation}.", 70, 16
        return f"Include equipment identification and symptoms in the report {citation}.", 65, 15


def main():
    manual = Path("data/maintenance_manual.txt").read_text(encoding="utf-8")
    chunks = [DocumentChunk("maintenance_manual.txt", 1, 0, manual)]
    retriever = VectorRetriever(DemoEmbedder())
    retriever.index(chunks)
    assistant = GroundedAssistant(retriever, DemoGenerator(), min_similarity=0.15)
    report = run_evaluation(assistant, "eval/questions.json", "eval/offline_results.json")
    print(json.dumps({
        "pass_rate": report["pass_rate"],
        "passed_cases": report["passed_cases"],
        "cases": report["cases"],
        "average_check_score": report["average_check_score"],
        "tokens": "Demo generator token counts are illustrative, not provider usage."
    }, indent=2))


if __name__ == "__main__":
    main()
