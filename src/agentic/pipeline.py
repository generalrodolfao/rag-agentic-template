"""
Stage 3 — Agentic RAG

Pipeline com agente autônomo (LangGraph):
  1. Planner — analisa a pergunta e cria um plano de busca
  2. Retrieval — executa buscas iterativas (pode buscar mais de uma rodada)
  3. Grader — avalia se os chunks recuperados são relevantes
  4. Reflection — verifica se a resposta final é completa e precisa
  5. Refinement — refina ou expande a resposta se necessário

Fluxo:
  Pergunta → Planner → Retrieval Loop (até N tentativas) → Grader →
    → Geração → Reflection → (Refinamento ou Final)
"""

from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from src.config import COLLECTION_NAME, PERSIST_DIR, TOP_K, LLM_MODEL

PLANNER_PROMPT = """Você é um planner de busca. Dada uma pergunta, crie um plano de busca com os tópicos que precisam ser pesquisados.

Pergunta: {question}

Retorne 2-4 tópicos de busca, um por linha, sem numeração:"""

GRADER_PROMPT = """Você é um grader que avalia se um trecho de documento é útil para responder uma pergunta.

Trecho: {chunk}

Pergunta: {question}

Retorne apenas "sim" ou "nao":"""

GENERATE_PROMPT = """Com base nos documentos recuperados, responda à pergunta do usuário.

Contexto:
{context}

Pergunta: {question}

Instruções:
- Seja completo e preciso
- Cites os trechos relevantes do contexto
- Se a informação for insuficiente, indique claramente o que falta
- Se aplicar, mencione limitações ou premissas

Resposta:"""

REFLECTION_PROMPT = """Analise a resposta gerada para a pergunta e identifique:

1. A resposta responde completamente a pergunta? (sim/não)
2. Há informações no contexto que não foram usadas? (liste)
3. A resposta contém alguma imprecisão? (detalhe)
4. O que poderia ser melhorado?

Pergunta: {question}
Resposta: {answer}
Contexto disponível: {context}

Retorne sua análise em tópicos:"""

REFINE_PROMPT = """Com base na análise de reflexão abaixo, refine sua resposta original para torná-la mais completa e precisa.

Pergunta: {question}
Resposta original: {answer}
Análise de reflexão: {reflection}
Contexto: {context}

Nova resposta refinada:"""

MAX_RETRIEVAL_ROUNDS = 3


class AgentState(TypedDict):
    question: str
    plan: list[str]
    chunks: Annotated[list[str], operator.add]
    current_topic_index: int
    retrieval_count: int
    answer: str
    reflection: str
    refined_answer: str
    iteration: int


class AgenticRAG:
    def __init__(self, model_name=LLM_MODEL):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR,
        )
        self.planner_prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
        self.grader_prompt = ChatPromptTemplate.from_template(GRADER_PROMPT)
        self.generate_prompt = ChatPromptTemplate.from_template(GENERATE_PROMPT)
        self.reflection_prompt = ChatPromptTemplate.from_template(REFLECTION_PROMPT)
        self.refine_prompt = ChatPromptTemplate.from_template(REFINE_PROMPT)
        self.graph = self._build_graph()

    def _plan(self, state: AgentState) -> dict:
        messages = self.planner_prompt.format_messages(question=state["question"])
        response = self.llm.invoke(messages)
        topics = [t.strip() for t in response.content.strip().split("\n") if t.strip()]
        return {
            "plan": topics[:4],
            "current_topic_index": 0,
            "retrieval_count": 0,
            "chunks": [],
            "iteration": 0,
        }

    def _retrieve(self, state: AgentState) -> dict:
        idx = state["current_topic_index"]
        if idx >= len(state["plan"]):
            return {"current_topic_index": idx}

        topic = state["plan"][idx]
        docs = self.store.similarity_search(topic, k=TOP_K)
        new_chunks = [doc.page_content for doc in docs]
        seen = set(state.get("chunks", []))
        unique_new = [c for c in new_chunks if c not in seen]

        return {
            "chunks": unique_new,
            "current_topic_index": idx + 1,
            "retrieval_count": state["retrieval_count"] + 1,
        }

    def _grade(self, state: AgentState) -> dict:
        relevant = []
        for chunk in state.get("chunks", []):
            messages = self.grader_prompt.format_messages(
                chunk=chunk[:1000], question=state["question"]
            )
            response = self.llm.invoke(messages)
            if "sim" in response.content.strip().lower():
                relevant.append(chunk)
        return {"chunks": relevant if relevant else state.get("chunks", [])}

    def _generate(self, state: AgentState) -> dict:
        context = "\n\n---\n\n".join(state.get("chunks", []))
        messages = self.generate_prompt.format_messages(
            context=context or "Nenhum documento recuperado.",
            question=state["question"],
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _reflect(self, state: AgentState) -> dict:
        context = "\n\n---\n\n".join(state.get("chunks", []))
        messages = self.reflection_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
            context=context or "Nenhum documento recuperado.",
        )
        response = self.llm.invoke(messages)
        return {"reflection": response.content}

    def _refine(self, state: AgentState) -> dict:
        context = "\n\n---\n\n".join(state.get("chunks", []))
        messages = self.refine_prompt.format_messages(
            question=state["question"],
            answer=state["answer"],
            reflection=state["reflection"],
            context=context or "Nenhum documento recuperado.",
        )
        response = self.llm.invoke(messages)
        return {"refined_answer": response.content}

    def _should_continue(self, state: AgentState) -> Literal["retrieve", "grade"]:
        if state["current_topic_index"] < len(state["plan"]):
            return "retrieve"
        return "grade"

    def _should_refine(self, state: AgentState) -> Literal["refine", "__end__"]:
        needs_refinement = (
            state.get("iteration", 0) < 1
            and state.get("reflection")
            and "não" in state["reflection"].lower()[:200]
        )
        if needs_refinement:
            return "refine"
        return END

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("planner", self._plan)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("grade", self._grade)
        builder.add_node("generate", self._generate)
        builder.add_node("reflect", self._reflect)
        builder.add_node("refine", self._refine)

        builder.set_entry_point("planner")
        builder.add_edge("planner", "retrieve")
        builder.add_conditional_edges("retrieve", self._should_continue)
        builder.add_edge("grade", "generate")
        builder.add_edge("generate", "reflect")
        builder.add_conditional_edges("reflect", self._should_refine)
        builder.add_edge("refine", END)
        builder.add_edge("grade", END)

        return builder.compile()

    def answer(self, query: str) -> dict:
        initial = AgentState(
            question=query,
            plan=[],
            chunks=[],
            current_topic_index=0,
            retrieval_count=0,
            answer="",
            reflection="",
            refined_answer="",
            iteration=0,
        )
        result = self.graph.invoke(initial)
        return {
            "query": query,
            "plan": result.get("plan", []),
            "chunks_retrieved": len(result.get("chunks", [])),
            "answer": result.get("refined_answer") or result.get("answer", ""),
            "reflection": result.get("reflection", ""),
        }
