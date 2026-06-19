from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import COLLECTION_NAME, PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def load_documents(doc_dir: str):
    loader = DirectoryLoader(doc_dir, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(doc_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load() + pdf_loader.load()
    return docs


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def get_vector_store(embedding_model=EMBEDDING_MODEL):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def ingest(doc_dir: str):
    docs = load_documents(doc_dir)
    chunks = split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Ingested {len(chunks)} chunks from {len(docs)} documents")
    return store
