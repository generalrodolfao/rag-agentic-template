"""
Stage 1 — Naive RAG

Pipeline linear:
  1. Retrieve chunks por similaridade de embedding
  2. Concatena chunks no prompt
  3. LLM gera resposta

Limitações (que os próximos stages resolvem):
  - Chunking fixo sem contexto semântico
  - Query é usada "como está" (sem transformação)
  - Busca exclusivamente vetorial (sem keyword)
  - Sem re-ranking — chunks podem ser irrelevantes
  - Uma única rodada — sem refinamento
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

from src.config import COLLECTION_NAME, PERSIST_DIR, TOP_K, LLM_MODEL

PROMPT_TEMPLATE = """Você é um assistente de IA especializado em responder perguntas com base em documentos fornecidos.

Contexto:
{context}

Pergunta: {question}

Instruções:
- Responda apenas com base no contexto acima
- Se o contexto não contiver informação suficiente, diga "Não encontrei essa informação nos documentos"
- Cites trechos relevantes quando possível

Resposta:"""


class NaiveRAG:
    def __init__(self, model_name=LLM_MODEL):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=PERSIST_DIR,
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        docs = self.store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def answer(self, query: str) -> dict:
        chunks = self.retrieve(query)
        context = "\n\n---\n\n".join(chunks)
        messages = self.prompt.format_messages(context=context, question=query)
        response = self.llm.invoke(messages)

        return {
            "query": query,
            "chunks_retrieved": len(chunks),
            "context": context,
            "answer": response.content,
        }
