# Evaluation

## What we measure and why

Two things distinguish a RAG system from a demo script: it retrieves the right
chunks reliably, and its answers are grounded in what was retrieved (not hallucinated).

We measure both:

| Metric | What it measures | How to run |
|---|---|---|
| **Recall@k** (k=3, 5) | Does at least one relevant chunk appear in the top-k results? | `make eval` |
| **MRR** | On average, how high is the first relevant chunk ranked? | `make eval` |
| **Faithfulness smoke tests** | Do fixed Q→A pairs still produce answers that cite the correct chunk? | `pytest` |

## Evaluation dataset

`eval/qa_set.jsonl` — a manually labeled set of question/relevant-chunk-id pairs,
constructed by:
1. Ingesting a sample document corpus into the vectorstore.
2. Writing 30–100 questions whose answers are unambiguously in one specific chunk.
3. Recording the chunk ID(s) for each question.

The labeled set lives in version control so results are reproducible.

## Results

Run `make eval` and fill in this table after each significant pipeline change:

| Pipeline variant | Recall@3 | Recall@5 | MRR | Notes |
|---|---|---|---|---|
| Baseline FAISS + all-MiniLM-L6-v2 | | | | |
| Groq production pipeline (llama3-8b) | | | | |

## Interpretation guidance

- **Recall@5 > 0.80** is a reasonable minimum bar for most use cases.
- **MRR < 0.50** means the right chunk is usually buried below position 2 — improve chunking or embedding model.
- If Recall@5 is high but end-to-end answer quality is poor, the bottleneck is generation (prompt, model) not retrieval.
