"""
Stage 2 — Advanced RAG

Pipeline com melhorias sobre o naive:
  1. Query Rewriting — reescreve a pergunta para melhor retrieval
  2. Query Decomposition — quebra perguntas complexas em sub-perguntas
  3. Hybrid Search — combina busca vetorial + keyword (BM25)
  4. Reciprocal Rank Fusion — combina rankings dos dois retrievers
  5. Re-ranking — cross-encoder para reposicionar chunks por relevância

"""

import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from src.config import COLLECTION_NAME, PERSIST_DIR, TOP_K, LLM_MODEL

REWRITE_PROMPT = """Dada a pergunta de um usuário, reescreva-a como uma pergunta autônoma e bem formulada para busca em documentos.

Pergunta original: {question}

Apenas retorne a pergunta reescrita, sem explicações:"""

DECOMPOSE_PROMPT = """A pergunta abaixo pode conter múltiplas intenções. Identifique e liste cada pergunta atômica que precisa ser respondida.

Pergunta: {question}

Retorne uma pergunta por linha, numerada:"""

RESPONSE_PROMPT = """Você é um assistente de IA especializado em responder perguntas com base em documentos.

Contexto:
{context}

Pergunta original: {question}

Instruções:
- Responda de forma completa e direta
- Cite os documentos/fontes quando possível
- Se o contexto for insuficiente, indique o que falta

Resposta:"""


class AdvancedRAG:
    def __init__(self, model_name=LLM_MODEL):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR,
        )
        self.rewrite_prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
        self.decompose_prompt = ChatPromptTemplate.from_template(DECOMPOSE_PROMPT)
        self.response_prompt = ChatPromptTemplate.from_template(RESPONSE_PROMPT)

    def _build_ensemble(self, chunks: list[str]):
        docs = [Document(page_content=c) for c in chunks]
        bm25 = BM25Retriever.from_documents(docs, k=TOP_K)
        vector = self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        return EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=[0.3, 0.7],
        )

    def _rewrite_query(self, query: str) -> str:
        messages = self.rewrite_prompt.format_messages(question=query)
        response = self.llm.invoke(messages)
        return response.content.strip()

    def _decompose_query(self, query: str) -> list[str]:
        messages = self.decompose_prompt.format_messages(question=query)
        response = self.llm.invoke(messages)
        lines = response.content.strip().split("\n")
        sub_questions = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned:
                sub_questions.append(cleaned)
        return sub_questions if len(sub_questions) > 1 else [query]

    def _rerank(self, query: str, chunks: list[str], top_k: int = TOP_K) -> list[str]:
        scored = []
        for chunk in chunks:
            relevance = self.llm.invoke(
                f"Classifique a relevância do trecho abaixo para responder: '{query}'\n\n"
                f"Trecho: {chunk[:500]}\n\n"
                f"Retorne apenas um número de 0 a 10:"
            )
            try:
                score = float(re.search(r"(\d+)", relevance.content).group(1))
            except (AttributeError, ValueError):
                score = 5.0
            scored.append((score, chunk))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored[:top_k]]

    def retrieve(self, query: str) -> list[str]:
        rewritten = self._rewrite_query(query)
        sub_questions = self._decompose_query(rewritten)
        all_chunks = []
        for sq in sub_questions:
            ensemble = self._build_ensemble([sq])
            docs = ensemble.invoke(sq)
            all_chunks.extend([d.page_content for d in docs])
        seen = set()
        unique = []
        for c in all_chunks:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return self._rerank(rewritten, unique)

    def answer(self, query: str) -> dict:
        chunks = self.retrieve(query)
        context = "\n\n---\n\n".join(chunks)
        messages = self.response_prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)

        return {
            "query": query,
            "chunks_retrieved": len(chunks),
            "context": context,
            "answer": response.content,
        }
