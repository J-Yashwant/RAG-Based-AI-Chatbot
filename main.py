import os
import csv
import shutil
import re
from typing import List

import pandas as pd
from rank_bm25 import BM25Okapi

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from bs4 import BeautifulSoup


# ==================================================
# RAG SYSTEM
# ==================================================
class UniversalRAG:
    def __init__(self, db_path="./chroma_db", model_name="qwen2.5:7b"):
        self.db_path = db_path

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        self.llm = ChatOllama(
            model=model_name,
            temperature=0
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        self.vector_store = None

        # CSV specific
        self.dataframes = {}          # filename → DataFrame
        self.bm25_indexes = {}        # filename → BM25
        self.bm25_docs = {}           # filename → text rows


# ==================================================
# CSV HANDLING (AUTO-SCHEMA)
# ==================================================
    def load_csv(self, path: str) -> List[Document]:
        filename = os.path.basename(path)

        df = pd.read_csv(path)
        self.dataframes[filename] = df

        docs = []
        texts = []

        for idx, row in df.iterrows():
            text = f"Row {idx+1}\n" + "\n".join(
                f"{col}: {row[col]}" for col in df.columns
            )

            texts.append(text)

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "row": idx + 1
                    }
                )
            )

        # Build BM25 index
        tokenized = [t.lower().split() for t in texts]
        self.bm25_indexes[filename] = BM25Okapi(tokenized)
        self.bm25_docs[filename] = docs

        return self.text_splitter.split_documents(docs)


# ==================================================
# OTHER LOADERS
# ==================================================
    def load_txt(self, path):
        docs = TextLoader(path).load()
        return self.text_splitter.split_documents(docs)

    def load_pdf(self, path):
        docs = PyPDFLoader(path).load()
        return self.text_splitter.split_documents(docs)

    def load_html(self, path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n")
        return self.text_splitter.split_documents([
            Document(page_content=text, metadata={"source": os.path.basename(path)})
        ])


# ==================================================
# INDEXING
# ==================================================
    def load_and_index(self, directory):
        all_docs = []

        print("\n--- 📂 Loading Documents ---")

        for file in os.listdir(directory):
            path = os.path.join(directory, file)

            try:
                if file.endswith(".csv"):
                    all_docs.extend(self.load_csv(path))
                    print(f"✅ CSV loaded: {file}")

                elif file.endswith(".pdf"):
                    all_docs.extend(self.load_pdf(path))
                    print(f"✅ PDF loaded: {file}")

                elif file.endswith(".txt"):
                    all_docs.extend(self.load_txt(path))
                    print(f"✅ TXT loaded: {file}")

                elif file.endswith(".html"):
                    all_docs.extend(self.load_html(path))
                    print(f"✅ HTML loaded: {file}")

            except Exception as e:
                print(f"❌ Error loading {file}: {e}")

        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        self.vector_store = Chroma.from_documents(
            all_docs,
            self.embeddings,
            persist_directory=self.db_path
        )

        print("--- 🚀 RAG SYSTEM READY ---")


# ==================================================
# 🔢 PANDAS EXECUTION (AUTO MATH)
# ==================================================
    def try_pandas_math(self, question: str):
        keywords = ["average", "sum", "total", "maximum", "minimum", "count", "mean"]

        if not any(k in question.lower() for k in keywords):
            return None

        for name, df in self.dataframes.items():
            for col in df.columns:
                if col.lower() in question.lower():
                    try:
                        if "average" in question or "mean" in question:
                            return f"Average {col}: {df[col].mean()}"
                        if "sum" in question or "total" in question:
                            return f"Total {col}: {df[col].sum()}"
                        if "maximum" in question:
                            return f"Maximum {col}: {df[col].max()}"
                        if "minimum" in question:
                            return f"Minimum {col}: {df[col].min()}"
                        if "count" in question:
                            return f"Count: {df[col].count()}"
                    except:
                        pass

        return None


# ==================================================
# 🔍 HYBRID SEARCH (BM25 + EMBEDDINGS)
# ==================================================
    def hybrid_search(self, question, k=6):
        docs = []

        # BM25
        for filename, bm25 in self.bm25_indexes.items():
            scores = bm25.get_scores(question.lower().split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
            docs.extend(self.bm25_docs[filename][i] for i in top_idx)

        # Embeddings
        docs.extend(
            self.vector_store.similarity_search(question, k=3)
        )

        # Deduplicate
        seen = set()
        unique = []
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("row"))
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique[:k]


# ==================================================
# QUERY
# ==================================================
    def get_answer(self, question: str):
        # 1️⃣ Try Pandas math
        math_answer = self.try_pandas_math(question)
        if math_answer:
            return math_answer, ["Computed using Pandas"]

        # 2️⃣ Hybrid retrieval
        docs = self.hybrid_search(question)

        if not docs:
            return "No relevant data found.", []

        context = "\n\n".join(d.page_content for d in docs)
        sources = {d.metadata.get("source") for d in docs}

        prompt = ChatPromptTemplate.from_template("""
You are a data analysis assistant.

Rules:
- Use ONLY the provided context
- Extract values exactly
- If information is missing, say "Not available in the data"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""")

        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return answer, sources


# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    rag = UniversalRAG()

    doc_dir = "./documents"
    if not os.path.exists(doc_dir):
        print("Create a 'documents' folder and add files.")
        exit()

    rag.load_and_index(doc_dir)

    while True:
        q = input("\nEnter Question (exit to quit): ")
        if q.lower() in ["exit", "quit"]:
            break

        ans, src = rag.get_answer(q)

        print("\n💡 ANSWER:\n", ans)
        print("\n--- 🔍 Source ---")
        for s in src:
            print("-", s)
