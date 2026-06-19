"""
Testes unitários para o RAG Agentic Template.

Nota: Os testes de pipeline requerem OPENAI_API_KEY e não rodam no CI
      sem a secret configurada. Os testes abaixo cobrem a parte isolável.
"""

from pathlib import Path
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


class TestConfig:
    def test_defaults(self):
        assert CHUNK_SIZE == 1000
        assert CHUNK_OVERLAP == 200
        assert TOP_K == 4


class TestLoader:
    def test_split_documents(self):
        from langchain_core.documents import Document
        from src.loader import split_documents

        docs = [Document(page_content="A " * 500 + "B " * 500)]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(len(c.page_content.split()) <= 220 for c in chunks)

    def test_empty_directory(self):
        from src.loader import load_documents

        empty_dir = Path("tests/fixtures/empty")
        empty_dir.mkdir(parents=True, exist_ok=True)
        docs = load_documents(str(empty_dir))
        assert len(docs) == 0


class TestNaivePipeline:
    def test_prompt_structure(self):
        from src.naive.pipeline import PROMPT_TEMPLATE

        assert "{context}" in PROMPT_TEMPLATE
        assert "{question}" in PROMPT_TEMPLATE

    def test_instantiation(self):
        from src.naive.pipeline import NaiveRAG

        rag = NaiveRAG(model_name="gpt-4o-mini")
        assert rag.llm is not None
        assert rag.prompt is not None


class TestAdvancedPipeline:
    def test_rewrite_prompt(self):
        from src.advanced.pipeline import REWRITE_PROMPT, DECOMPOSE_PROMPT, RESPONSE_PROMPT

        assert "{question}" in REWRITE_PROMPT
        assert "{question}" in DECOMPOSE_PROMPT
        assert "{context}" in RESPONSE_PROMPT
        assert "{question}" in RESPONSE_PROMPT

    def test_instantiation(self):
        from src.advanced.pipeline import AdvancedRAG

        rag = AdvancedRAG(model_name="gpt-4o-mini")
        assert rag.llm is not None
        assert rag.rewrite_prompt is not None


class TestAgenticPipeline:
    def test_prompts(self):
        from src.agentic.pipeline import (
            PLANNER_PROMPT,
            GRADER_PROMPT,
            GENERATE_PROMPT,
            REFLECTION_PROMPT,
            REFINE_PROMPT,
        )

        assert "{question}" in PLANNER_PROMPT
        assert "{chunk}" in GRADER_PROMPT
        assert "{question}" in GRADER_PROMPT
        assert "{context}" in GENERATE_PROMPT
        assert "{question}" in GENERATE_PROMPT
        assert "{answer}" in REFLECTION_PROMPT
        assert "{reflection}" in REFINE_PROMPT

    def test_instantiation(self):
        from src.agentic.pipeline import AgenticRAG

        rag = AgenticRAG(model_name="gpt-4o-mini")
        assert rag.llm is not None
        assert rag.graph is not None

    def test_state_type(self):
        from src.agentic.pipeline import AgentState

        state = AgentState(
            question="teste",
            plan=["topic1"],
            chunks=["chunk1"],
            current_topic_index=0,
            retrieval_count=0,
            answer="resposta",
            reflection="reflexão",
            refined_answer="",
            iteration=0,
        )
        assert state["question"] == "teste"
        assert state["plan"] == ["topic1"]


class TestObservability:
    def test_import(self):
        from src.observability import get_langfuse, trace_rag, log_retrieval

        assert callable(trace_rag)
        assert callable(log_retrieval)


class TestQuickstart:
    def test_sample_data_structure(self):
        from examples.quickstart import SAMPLE_DATA, SAMPLE_QUESTIONS

        assert len(SAMPLE_DATA) > 100
        assert len(SAMPLE_QUESTIONS) == 3
        assert "alucinações" in SAMPLE_DATA
        assert "LangGraph" in SAMPLE_DATA
