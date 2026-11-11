from __future__ import annotations
import json
from pathlib import Path
from .core import GroundedAssistant


def run_evaluation(assistant: GroundedAssistant, dataset_path: str, output_path: str) -> dict:
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    results = []
    passed = 0
    total_cost = 0.0
    total_input = total_output = 0

    for case in cases:
        response = assistant.ask(case["question"])
        checks = {
            "expected_status": response.status == case["expected_status"],
            "expected_contains": all(
                phrase.lower() in response.answer.lower()
                for phrase in case.get("answer_contains", [])
            ),
            "must_have_sources": (not case.get("must_have_sources", False)
                                  or bool(response.retrieved_passages)),
            "must_not_call_llm": (not case.get("must_not_call_llm", False)
                                  or response.input_tokens == 0 and response.output_tokens == 0),
        }
        score = sum(checks.values()) / len(checks)
        passed += int(all(checks.values()))
        total_cost += response.estimated_cost_usd
        total_input += response.input_tokens
        total_output += response.output_tokens
        results.append({
            "id": case["id"], "question": case["question"],
            "status": response.status, "answer": response.answer,
            "checks": checks, "score": round(score, 2),
            "citations": response.citations,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "estimated_cost_usd": round(response.estimated_cost_usd, 8),
        })

    report = {
        "cases": len(cases), "passed_cases": passed,
        "pass_rate": round(passed / len(cases), 4) if cases else 0,
        "average_check_score": round(sum(r["score"] for r in results) / len(results), 4) if results else 0,
        "total_input_tokens": total_input, "total_output_tokens": total_output,
        "estimated_total_cost_usd": round(total_cost, 8),
        "results": results,
    }
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
