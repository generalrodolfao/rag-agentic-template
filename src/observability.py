"""
Observabilidade com LangFuse

Fornece decorators e utilitários para tracing dos pipelines de RAG.

Uso:
  from src.observability import trace_rag, get_langfuse

  @trace_rag
  def answer(self, query):
      ...
"""

import os
from functools import wraps
from langfuse import LangFuse
from langfuse.decorators import langfuse_context, observe

_langfuse = None


def get_langfuse() -> LangFuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = LangFuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    return _langfuse


def trace_rag(func):
    @wraps(func)
    @observe(as_type="generation")
    def wrapper(*args, **kwargs):
        langfuse_context.update_current_trace(
            name=f"rag_{func.__name__}",
            session_id=kwargs.get("query", str(args[1] if len(args) > 1 else "")),
            metadata={"stage": func.__module__.split(".")[-2]},
        )
        result = func(*args, **kwargs)
        langfuse_context.update_current_observation(
            output=result.get("answer", ""),
            usage={
                "input": [result.get("query", "")],
                "output": [result.get("answer", "")],
            },
        )
        return result

    return wrapper


def log_retrieval(query: str, chunks: list[str], stage: str):
    langfuse = get_langfuse()
    trace = langfuse.trace(
        name=f"retrieval_{stage}",
        input={"query": query, "stage": stage},
        output={"chunks_count": len(chunks)},
        metadata={
            "chunk_lengths": [len(c) for c in chunks],
            "total_chars": sum(len(c) for c in chunks),
        },
    )
    return trace


def log_evaluation(scores: dict, stage: str):
    langfuse = get_langfuse()
    trace = langfuse.trace(
        name=f"evaluation_{stage}",
        metadata={"scores": scores},
    )
    return trace
