"""
01_min_rag_cli.py — Minimal RAG pipeline (CLI)

Usage:
    python examples/01_min_rag_cli.py --query "What does the document say about X?"
    python examples/01_min_rag_cli.py --ingest path/to/docs/ --query "..."

This is the entry point to verify your environment works end-to-end:
  ingest documents → embed → store in FAISS → retrieve → generate answer with Groq.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
DOCS_PATH = os.getenv("DOCS_PATH", "./data/docs")


def build_or_load_index(docs_path: str, index_path: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    index_dir = Path(index_path)

    if index_dir.exists():
        print(f"Loading existing index from {index_path}")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print(f"Building index from documents in {docs_path} ...")
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No .txt documents found in {docs_path}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  Indexing {len(chunks)} chunks from {len(docs)} documents")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_path)
    print(f"  Index saved to {index_path}")
    return vectorstore


def run(query: str, docs_path: str = DOCS_PATH, index_path: str = INDEX_PATH):
    vectorstore = build_or_load_index(docs_path, index_path)
    llm = ChatGroq(model=GROQ_MODEL, groq_api_key=os.environ["GROQ_API_KEY"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        return_source_documents=True,
    )
    result = chain.invoke({"query": query})
    print("\n--- Answer ---")
    print(result["result"])
    print("\n--- Sources ---")
    for doc in result["source_documents"]:
        print(f"  [{doc.metadata.get('source', 'unknown')}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--ingest", default=DOCS_PATH)
    parser.add_argument("--index", default=INDEX_PATH)
    args = parser.parse_args()
    run(args.query, docs_path=args.ingest, index_path=args.index)
