"""
Evaluación com RAGAS

Métricas calculadas:
  - Faithfulness: a resposta é fiel ao contexto?
  - Answer Relevancy: a resposta é relevante à pergunta?
  - Context Precision: o contexto contém apenas informação relevante?
  - Context Recall: o contexto cobre tudo que é necessário?
  - Answer Correctness: a resposta está factualmente correta (se ground truth disponível)

Uso:
  python -m evaluation.run_ragas
"""

import os
import json
from pathlib import Path

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm as ragas_llm
from ragas.embeddings import embedding as ragas_embeddings
from datasets import Dataset


def prepare_dataset(answers: list[dict]) -> Dataset:
    rows = []
    for a in answers:
        rows.append({
            "question": a["query"],
            "answer": a["answer"],
            "contexts": [a.get("context", "")],
        })
    return Dataset.from_list(rows)


def run_evaluation(answers: list[dict]) -> dict:
    dataset = prepare_dataset(answers)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores = {}
    for metric, value in result.items():
        if isinstance(value, (int, float)):
            scores[metric] = round(value, 4)
        else:
            try:
                scores[metric] = round(float(value), 4)
            except (TypeError, ValueError):
                scores[metric] = str(value)

    return scores


def print_report(scores: dict):
    print("\n" + "=" * 50)
    print("📊 RAGAS Evaluation Report")
    print("=" * 50)
    for metric, value in scores.items():
        bar = "█" * int(float(value) * 20) if isinstance(value, (int, float)) else "N/A"
        print(f"  {metric:25s} {value:.4f}  {bar}")
    print("=" * 50)


if __name__ == "__main__":
    results_path = Path("evaluation/results.json")
    if results_path.exists():
        with open(results_path) as f:
            answers = json.load(f)
    else:
        print("Nenhum resultado encontrado. Execute os pipelines primeiro:")
        print("  python -m src.naive.pipeline")
        print("  python -m src.advanced.pipeline")
        print("  python -m src.agentic.pipeline")
        exit(1)

    scores = run_evaluation(answers)
    print_report(scores)

    output_path = Path("evaluation/scores.json")
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores salvos em {output_path}")
