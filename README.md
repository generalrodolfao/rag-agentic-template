<h1 align="center">RAG Agentic Template</h1>

<p align="center">
  <em>Naive → Advanced → Agentic — A evolução prática do RAG em produção</em>
  <br/>
  <a href="https://github.com/generalrodolfao/rag-agentic-template"><img src="https://img.shields.io/badge/status-production--ready-00154E?style=flat-square" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-00154E?style=flat-square" /></a>
  <a href="https://www.langfuse.com"><img src="https://img.shields.io/badge/observability-LangFuse-00154E?style=flat-square" /></a>
  <a href="https://docs.ragas.io"><img src="https://img.shields.io/badge/evaluation-RAGAS-00154E?style=flat-square" /></a>
</p>

---

## Visão Geral

Template progressivo de RAG que evolui do básico ao avançado, mostrando **como cada camada de complexidade resolve problemas específicos** da geração aumentada por recuperação.

Cada estágio é autocontido e pode ser executado independentemente — ideal para estudos, benchmarks e como ponto de partida para projetos reais.

---

## Arquitetura

```
                    ┌──────────────────────────────────────────────────┐
                    │                  Pergunta                        │
                    └──────────┬───────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                     ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
   │  Naive RAG  │    │ Advanced RAG │    │   Agentic RAG    │
   │             │    │              │    │                  │
   │  Retrieve   │    │  Rewrite     │    │  Planner         │
   │     ↓       │    │     ↓        │    │     ↓            │
   │  Generate   │    │  Decompose   │    │  Retrieve Loop   │
   │             │    │     ↓        │    │     ↓            │
   │  Linear     │    │  Hybrid      │    │  Grader          │
   │  pipeline   │    │  Search      │    │     ↓            │
   │             │    │     ↓        │    │  Generate        │
   │             │    │  Rerank      │    │     ↓            │
   │             │    │     ↓        │    │  Reflect         │
   │             │    │  Generate    │    │     ↓            │
   │             │    │              │    │  Refine          │
   └─────────────┘    └──────────────┘    └──────────────────┘
          │                    │                     │
          └────────────────────┼─────────────────────┘
                               ▼
                    ┌──────────────────┐
                    │   LangFuse       │ ← Tracing & Observability
                    │   + RAGAS        │ ← Evaluation Metrics
                    └──────────────────┘
```

---

## Estágios

### 1. Naive RAG — `src/naive/pipeline.py`

Pipeline linear clássico. Ideal para entender os fundamentos.

| Etapa | Descrição |
|---|---|
| Chunking | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| Embedding | OpenAI text-embedding-3-small |
| Retrieval | Similaridade por cosseno no ChromaDB |
| Geração | Prompt único com contexto concatenado |

**Limitações:** chunk fixo, query sem transformação, sem reranking, sem refinamento.

### 2. Advanced RAG — `src/advanced/pipeline.py`

Oito melhorias incrementais sobre o naive.

| Etapa | Descrição | Problema que resolve |
|---|---|---|
| Query Rewriting | Reescreve a pergunta para melhor match semântico | Queries mal formuladas |
| Query Decomposition | Quebra perguntas complexas em sub-perguntas | Perguntas multi-intenção |
| Hybrid Search | Vector + BM25 (keyword) | Termos raros vs semântica |
| RRF (Fusion) | Combina rankings dos dois retrievers | Viés de cada método |
| Cross-encoder Rerank | Re-classifica chunks por relevância | Falsos positivos da相似idade |

### 3. Agentic RAG — `src/agentic/pipeline.py`

Agente autônomo com LangGraph. O sistema decide **como, quando e se** deve buscar mais informação.

| Componente | Função |
|---|---|
| **Planner** | Analisa a pergunta e cria um plano de busca com 2-4 tópicos |
| **Retrieve Loop** | Executa uma busca por tópico do plano |
| **Grader** | Filtra chunks irrelevantes antes da geração |
| **Generator** | Produz a resposta inicial com contexto |
| **Reflector** | Auto-avalia a resposta (completude, precisão) |
| **Refiner** | Refina a resposta com base na reflexão |

---

## Comparação

| Característica | Naive | Advanced | Agentic |
|---|---|---|---|
| Transformação de query | — | ✅ Rewrite + Decompose | ✅ Planner |
| Busca | Vetorial | Híbrida (vector + BM25) | Híbrida + iterativa |
| Reranking | — | ✅ Cross-encoder | ✅ Grader |
| Múltiplas rodadas | — | — | ✅ Até 3 tópicos |
| Reflexão | — | — | ✅ Auto-avaliação |
| Refinamento | — | — | ✅ Iterativo |
| Complexidade | Baixa | Média | Alta |
| Precisão típica | ~60-70% | ~75-85% | ~85-95% |

---

## Quickstart

```bash
# Clone
git clone https://github.com/generalrodolfao/rag-agentic-template.git
cd rag-agentic-template

# Configure
cp .env.example .env
# Edite .env com suas chaves (OPENAI_API_KEY)

# Instale
pip install -r requirements.txt

# Ingestão dos dados de exemplo (primeira vez)
python examples/quickstart.py --ingest

# Execute todos os estágios
python examples/quickstart.py
```

O script compara os 3 estágios lado a lado com as mesmas perguntas e salva os resultados em `evaluation/results.json`.

---

## Avaliação

```bash
# Execute a avaliação com RAGAS
python -m evaluation.run_ragas
```

Métricas calculadas:

| Métrica | Descrição |
|---|---|
| **Faithfulness** | A resposta é fiel ao contexto fornecido? |
| **Answer Relevancy** | A resposta é pertinente à pergunta? |
| **Context Precision** | O contexto contém apenas informação relevante? |
| **Context Recall** | O contexto cobre toda a informação necessária? |

---

## Observabilidade

Tracing via LangFuse cobre:
- Cada etapa do pipeline (retrieval, geração, reflexão)
- Métricas de latência e custo por query
- Comparação entre estágios na mesma sessão

Para ativar, configure as variáveis `LANGFUSE_*` no `.env`.

---

## Estrutura

```
rag-agentic-template/
├── src/
│   ├── naive/pipeline.py       # Stage 1 — RAG linear
│   ├── advanced/pipeline.py    # Stage 2 — Query transform + hybrid + rerank
│   ├── agentic/pipeline.py     # Stage 3 — LangGraph multi-step
│   ├── loader.py               # Ingestão de documentos
│   ├── config.py               # Configurações centralizadas
│   └── observability.py        # LangFuse tracing
├── evaluation/
│   └── run_ragas.py            # RAGAS evaluation
├── examples/
│   └── quickstart.py           # Comparação dos 3 estágios
├── data/sample_docs/           # Documentos de exemplo
├── .env.example
├── .gitignore
└── requirements.txt
```

---

<p align="center">
  <b>Data Squad</b> · <a href="https://dtsqd.com">dtsqd.com</a> · <a href="mailto:rodolfo@dtsqd.com">rodolfo@dtsqd.com</a>
  <br/>
  <a href="https://github.com/generalrodolfao"><img src="https://img.shields.io/badge/github-generalrodolfao-00154E?style=flat-square&logo=github&logoColor=F17405" /></a>
  <a href="https://linkedin.com/in/generalrodolfao"><img src="https://img.shields.io/badge/LinkedIn-generalrodolfao-00154E?style=flat-square&logo=linkedin&logoColor=F17405" /></a>
</p>
