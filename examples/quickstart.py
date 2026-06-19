"""
Quickstart — RAG Agentic Template

Uso:
  1. Configure .env (copie .env.example)
  2. python examples/quickstart.py --ingest   # apenas na primeira vez
  3. python examples/quickstart.py             # executa todos os pipelines

Compara os 3 estágios lado a lado na mesma pergunta.
"""

import argparse
import json
from pathlib import Path

from src.config import SAMPLE_DATA_DIR
from src.loader import ingest
from src.naive.pipeline import NaiveRAG
from src.advanced.pipeline import AdvancedRAG
from src.agentic.pipeline import AgenticRAG

SAMPLE_QUESTIONS = [
    "Quais são as principais estratégias para reduzir alucinações em LLMs?",
    "Como implementar um sistema multi-agente com LangGraph?",
    "Explique as diferenças entre RAG naive, advanced e agentic.",
]

SAMPLE_DATA = """# Redução de Alucinações em LLMs

## Técnicas Principais

1. **RAG (Retrieval-Augmented Generation)**: Conectar o LLM a fontes externas de dados
   reduz alucinações ao fornecer contexto factual para a geração.

2. **Prompt Engineering**: Técnicas como chain-of-thought, few-shot e system prompts
   bem estruturados reduzem significativamente a taxa de alucinação.

3. **Fine-tuning com Dados Curados**: Modelos ajustados com dados verificados
   apresentam até 40% menos alucinações segundo estudos recentes.

4. **Validação Pós-Geração**: Sistemas de verificação factual e auto-reflexão
   (como reflexion agents) capturam inconsistências antes da entrega.

## Métricas de Avaliação
- Faithfulness: a resposta é fiel ao contexto fornecido?
- Answer Relevancy: a resposta é pertinente à pergunta?
- Context Precision: o contexto usado é relevante?

# Sistemas Multi-Agente com LangGraph

## Arquitetura
LangGraph permite criar grafos de agentes onde cada nó é uma função
e as arestas definem o fluxo de controle entre agentes.

## Componentes
- **StateGraph**: grafo que mantém estado compartilhado entre nós
- **Nodes**: funções que processam e transformam o estado
- **Edges**: conexões condicionais ou fixas entre nós
- **Checkpointing**: persistência do estado entre execuções

## Padrões Comuns
1. Supervisor: um agente coordena outros especialistas
2. Sequencial: pipeline de agentes em série
3. Debate: múltiplos agentes discutem até consenso
4. Reflexão: agente gera, avalia e refina a própria saída

# Evolução do RAG

## RAG Naive
Pipeline linear: chunk → embedding → retrieve → generate.
Limitações: chunk fixo, sem reranking, sem transformação de query.

## RAG Advanced
Melhorias: query rewriting, decomposição, hybrid search, reranking.
Resultado: retrieval mais preciso e contextos mais relevantes.

## RAG Agentic
Inovações: planejamento multi-step, reflexão, refinamento iterativo.
Diferencial: o sistema decide quando e como buscar mais informação.
"""


def setup_sample_data():
    data_dir = Path(SAMPLE_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    sample_file = data_dir / "sample_data.txt"
    sample_file.write_text(SAMPLE_DATA)
    print(f"Dados de exemplo criados em {sample_file}")


def run_comparison(questions: list[str]):
    naive = NaiveRAG()
    advanced = AdvancedRAG()
    agentic = AgenticRAG()

    results = {"naive": [], "advanced": [], "agentic": []}

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Pergunta: {q}")
        print(f"{'='*60}")

        print("\n▶ Naive RAG:")
        r1 = naive.answer(q)
        print(f"  Chunks: {r1['chunks_retrieved']}")
        print(f"  Resposta: {r1['answer'][:200]}...")
        results["naive"].append(r1)

        print("\n▶ Advanced RAG:")
        r2 = advanced.answer(q)
        print(f"  Chunks: {r2['chunks_retrieved']}")
        print(f"  Resposta: {r2['answer'][:200]}...")
        results["advanced"].append(r2)

        print("\n▶ Agentic RAG:")
        r3 = agentic.answer(q)
        print(f"  Chunks: {r3['chunks_retrieved']}")
        print(f"  Plano: {r3.get('plan', [])}")
        print(f"  Resposta: {r3['answer'][:200]}...")
        results["agentic"].append(r3)

    output_path = Path("evaluation/results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true", help="Ingerir dados de exemplo")
    parser.add_argument("--questions", nargs="+", default=None, help="Perguntas customizadas")
    args = parser.parse_args()

    setup_sample_data()

    if args.ingest:
        ingest(SAMPLE_DATA_DIR)
        print("✅ Dados ingeridos. Agora execute sem --ingest para testar os pipelines.")

    questions = args.questions or SAMPLE_QUESTIONS
    run_comparison(questions)
