# Task 04 — Capstone: Grounded Industrial Maintenance Assistant

A document-grounded assistant that stays within an industrial maintenance remit, refuses unsafe/off-topic requests, retrieves evidence from local documents, cites sources, and measures token usage and estimated LLM cost.

## Features mapped to the brief

- **Guardrails:** blocks unsafe requests (e.g. bypassing interlocks) and off-topic questions before any LLM call.
- **RAG:** chunks and embeds local `.txt`/`.md` documents, retrieves relevant passages using cosine similarity, and displays the evidence.
- **Grounding:** the LLM is instructed to answer only from retrieved passages and to abstain when the answer is absent.
- **Evaluation:** a six-case JSON evaluation set checks expected status, answer phrases, source retrieval, and that blocked/absent questions do not call the model.
- **Usage and cost:** captures provider-reported input/output tokens and estimates cost using configurable per-million-token rates.

## Setup

Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your OpenAI API key. Never commit `.env`.

```env
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```

## Run the app

```bash
streamlit run app.py
```

The included `data/maintenance_manual.txt` is indexed by default. Ask questions about the manual in the UI.

## Run automated tests (no API key required)

```bash
pytest -q
```

Tests use deterministic fake embeddings and a fake generator. They verify guardrail blocking, retrieval, abstention, citations, and token/cost accounting.

## Run the offline evaluation (no API key required)

```bash
python eval/run_offline_eval.py
```

This executes the six fixed cases using deterministic demo embeddings and a demo generator. It verifies evaluation scoring and guardrail behavior, but its answer quality and token counts are not live-LLM results. The generated report is `eval/offline_results.json`.

## Run the live evaluation

```bash
python run_eval.py
```

This uses the live embedding model and OpenAI API, then writes `eval/results.json`. API usage and cost are measured from provider-reported token counts. The cost rates in `GroundedAssistant` default to illustrative rates and should be updated to match the model and current provider pricing before reporting financial estimates.

## Evaluation dataset

`eval/questions.json` contains six fixed cases:
1. Answerable pump maintenance question.
2. Answerable compressor measurement question.
3. Missing compressor specification (must abstain).
4. Unsafe interlock-bypass request (must block).
5. Off-topic request (must block).
6. Answerable maintenance-report question.

The evaluation script scores four checks per case and reports pass rate, average check score, tokens, and estimated cost. The checked phrase tests are intentionally simple and should be expanded for a production benchmark.

## Guardrail design and limitations

The first-line guardrail is a transparent keyword/pattern policy. It is intentionally small and inspectable for this capstone; it is not a comprehensive safety classifier and can miss paraphrases or block benign wording. Production use should add a dedicated classifier, adversarial tests, human review, and site-specific policy. Retrieved documents are untrusted evidence and cannot override system instructions. The system should not be used as the sole basis for safety-critical maintenance decisions.

## Cost calculation

Estimated cost:

`(input_tokens * input_price_per_1m + output_tokens * output_price_per_1m) / 1,000,000`

The defaults are illustrative. Set rates to the actual model pricing in effect when you run the evaluation. Embedding API cost is not included because the sample embedding model runs locally; local compute cost is also not estimated.

## Project structure

```text
task04-grounded-assistant/
├── app.py
├── run_eval.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── maintenance_manual.txt
├── eval/
│   └── questions.json
├── grounded_assistant/
│   ├── __init__.py
│   ├── core.py
│   └── evaluation.py
└── tests/
    └── test_core.py
```

## Submission checklist

- [x] Guardrails for unsafe and off-topic requests
- [x] Document retrieval and source passages
- [x] Honest not-in-documents response
- [x] Fixed evaluation dataset
- [x] Scoring script and report output
- [x] Token usage and estimated cost tracking
- [x] README and run instructions
- [ ] Run tests locally
- [ ] Run live evaluation and inspect `eval/results.json`
- [ ] Update pricing assumptions for the selected model
- [ ] Push to a public GitHub repository and submit its URL
