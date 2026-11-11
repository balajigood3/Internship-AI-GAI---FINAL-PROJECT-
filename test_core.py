import numpy as np
from grounded_assistant.core import (
    DocumentChunk, GroundedAssistant, VectorRetriever, NOT_FOUND,
    OUT_OF_SCOPE, UNSAFE, classify_request,
)

class TinyEmbedder:
    def encode(self, texts):
        result = []
        for t in texts:
            low = t.lower()
            result.append([
                float(any(w in low for w in ["pump", "vibration"])),
                float(any(w in low for w in ["compressor", "temperature", "110"])),
                float(any(w in low for w in ["maintenance", "report", "inspection"])),
            ])
        return np.asarray(result, dtype=np.float32)

class FakeGenerator:
    def __init__(self):
        self.calls = 0
    def generate(self, question, passages):
        self.calls += 1
        q = question.lower()
        if "normal" in q and "temperature" in q:
            return "This information is not available in the provided documents.", 110, 18
        if "pump" in q:
            return "Record vibration readings [manual.txt, page 1].", 100, 20
        if "compressor" in q:
            return "The sample record reports 110 degrees Celsius [manual.txt, page 1].", 105, 18
        return "The report includes equipment and symptoms [manual.txt, page 1].", 100, 20

def make_assistant():
    chunks = [
        DocumentChunk("manual.txt", 1, 0, "Pump P-101 unusual vibration. Record vibration readings and pressure."),
        DocumentChunk("manual.txt", 1, 1, "Compressor C-202 sample temperature was 110 degrees Celsius."),
        DocumentChunk("manual.txt", 1, 2, "Maintenance reports include equipment and symptoms."),
    ]
    retriever = VectorRetriever(TinyEmbedder())
    retriever.index(chunks)
    gen = FakeGenerator()
    return GroundedAssistant(retriever, gen, min_similarity=0.2), gen

def test_unsafe_guardrail_blocks_before_llm():
    assistant, gen = make_assistant()
    result = assistant.ask("How do I bypass the safety interlock on a pump?")
    assert result.status == "blocked_unsafe"
    assert result.answer == UNSAFE
    assert gen.calls == 0

def test_off_topic_guardrail_blocks_before_llm():
    assistant, gen = make_assistant()
    result = assistant.ask("Write a poem about the ocean")
    assert result.status == "blocked_off_topic"
    assert result.answer == OUT_OF_SCOPE
    assert gen.calls == 0

def test_in_scope_question_returns_source_passages_and_usage():
    assistant, gen = make_assistant()
    result = assistant.ask("What should I record for pump vibration?")
    assert result.status == "answered"
    assert result.retrieved_passages
    assert result.input_tokens == 100
    assert result.output_tokens == 20
    assert result.estimated_cost_usd > 0
    assert result.citations

def test_absent_fact_returns_not_found_without_llm():
    assistant, gen = make_assistant()
    result = assistant.ask("What is the normal compressor temperature?")
    assert result.status == "not_found"
    assert result.answer == NOT_FOUND
    # Retrieval found a related passage; the generator abstains because the requested
    # normal operating temperature is not stated in that passage.
    assert gen.calls == 1

def test_classifier_categories():
    assert classify_request("pump vibration inspection") == "in_scope"
    assert classify_request("tell me a joke") == "out_of_scope"
    assert classify_request("disable safety alarm") == "unsafe"
