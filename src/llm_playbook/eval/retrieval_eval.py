"""
Retrieval evaluation harness for llm-playbook.

Usage:
    python -m llm_playbook.eval.retrieval_eval --dataset eval/qa_set.jsonl --k 3 5

Expects a JSONL file where each line is:
    {"question": "...", "relevant_doc_ids": ["doc_012", "doc_045"]}

Reports Recall@k and Mean Reciprocal Rank (MRR) for a retriever function
you plug in below. This is intentionally minimal — the point is a
reproducible, checked-in number, not a research-grade benchmark.
"""

import argparse
import json
from dataclasses import dataclass


@dataclass
class EvalResult:
    recall_at_k: dict
    mrr: float
    n_queries: int


def load_qa_set(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve(query: str, k: int) -> list[str]:
    """
    Plug in your actual retriever here (FAISS / hybrid / whatever
    pipeline you're evaluating). Must return a list of doc_ids,
    ranked most-relevant first, length <= k.
    """
    raise NotImplementedError("Wire this up to your vectorstore.retrieval module")


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    top_k = set(retrieved[:k])
    hit = len(top_k & set(relevant)) > 0
    return 1.0 if hit else 0.0


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def run_eval(dataset_path: str, k_values: list[int]) -> EvalResult:
    qa_set = load_qa_set(dataset_path)
    recall_scores = {k: [] for k in k_values}
    rr_scores = []

    for item in qa_set:
        question = item["question"]
        relevant = item["relevant_doc_ids"]
        retrieved = retrieve(question, k=max(k_values))

        for k in k_values:
            recall_scores[k].append(recall_at_k(retrieved, relevant, k))
        rr_scores.append(reciprocal_rank(retrieved, relevant))

    recall_at_k_avg = {k: sum(v) / len(v) for k, v in recall_scores.items()}
    mrr = sum(rr_scores) / len(rr_scores) if rr_scores else 0.0

    return EvalResult(recall_at_k=recall_at_k_avg, mrr=mrr, n_queries=len(qa_set))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5])
    args = parser.parse_args()

    result = run_eval(args.dataset, args.k)

    print(f"Evaluated on {result.n_queries} queries")
    for k, score in result.recall_at_k.items():
        print(f"Recall@{k}: {score:.3f}")
    print(f"MRR: {result.mrr:.3f}")


if __name__ == "__main__":
    main()
