import os
import shutil
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ==========================================
# CORE RAG CLASS
# ==========================================
class UniversalRAG:
    def __init__(self, db_path="./chroma_db", model_name="qwen2.5:7b"):
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatOllama(model=model_name, temperature=0)
        
        self.standard_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=250, separators=["\n\n", "\n", ".", " "]
        )

        self.vector_store = None
        self.bm25_indexes = {}
        self.bm25_docs = {}

    def load_csv(self, path):
        filename = os.path.basename(path)
        df = pd.read_csv(path).fillna("N/A")
        docs, tokenized_corpus = [], []
        for idx, row in df.iterrows():
            content = f"File: {filename} | Data: " + " | ".join([f"{c}: {v}" for c, v in row.items()])
            doc = Document(page_content=content, metadata={"source": filename})
            docs.append(doc)
            tokenized_corpus.append(content.lower().split())
        self.bm25_indexes[filename] = BM25Okapi(tokenized_corpus)
        self.bm25_docs[filename] = docs
        return docs

    def load_and_index(self, directory):
        if not os.path.exists(directory): os.makedirs(directory)
        all_docs = []
        for file in os.listdir(directory):
            path = os.path.join(directory, file)
            try:
                if file.endswith(".pdf"):
                    loaded = PyPDFLoader(path).load_and_split(self.pdf_splitter)
                elif file.endswith(".csv"):
                    loaded = self.load_csv(path)
                elif file.endswith(".txt"):
                    loaded = TextLoader(path).load_and_split(self.standard_splitter)
                elif file.endswith(".html"):
                    with open(path, encoding="utf-8") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                        for s in soup(["script", "style"]): s.decompose()
                        text = soup.get_text(separator="\n")
                        loaded = self.standard_splitter.create_documents([text], metadatas=[{"source": file}])
                
                # Metadata Injection: Ensure the filename/source info is in every chunk
                for d in loaded:
                    d.page_content = f"[Document: {file}]\n{d.page_content}"
                all_docs.extend(loaded)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
        
        if os.path.exists(self.db_path): shutil.rmtree(self.db_path)
        self.vector_store = Chroma.from_documents(all_docs, self.embeddings, persist_directory=self.db_path)

    def search(self, question):
        # Increased k to 8 for better coverage of specific details like authors
        semantic_results = self.vector_store.similarity_search(question, k=8)
        csv_results = []
        for filename, bm25 in self.bm25_indexes.items():
            scores = bm25.get_scores(question.lower().split())
            if any(s > 0 for s in scores):
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
                csv_results.extend([self.bm25_docs[filename][i] for i in top_idx])
        unique = {d.page_content: d for d in (csv_results + semantic_results)}
        return list(unique.values())

    def get_answer(self, question):
        docs = self.search(question)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_template("""
        You are a highly precise document analysis expert. Use the following context to answer the user's question.
        
        Rules:
        1. If the information (like an author, date, or specific ID) is in the context, provide it clearly.
        2. If it is NOT in the context, say: "I'm sorry, I could not find information regarding that in the uploaded documents."
        3. Do not use outside knowledge.
        4. Do not mention the phrase "the context" or "the provided text" in your final answer. Just answer.

        Context: {context}
        Question: {question}
        Answer:
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})

# ==========================================
# ENHANCED STREAMLIT UI
# ==========================================
st.set_page_config(page_title="DocIntel AI", layout="wide")

# Custom CSS for a better UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    .stSidebar { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    .stTitle { color: #1e3a8a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

if "rag" not in st.session_state:
    st.session_state.rag = UniversalRAG()
    with st.status("Initializing Knowledge Base...", expanded=True) as status:
        st.write("Loading documents...")
        st.session_state.rag.load_and_index("./documents")
        status.update(label="System Ready!", state="complete", expanded=False)

st.title("📂 DocIntel AI")
st.subheader("Technical Document & Data Intelligence")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.header("Control Panel")
    st.info("The system is currently scanning your `./documents` folder.")
    if st.button("🔄 Refresh Knowledge Base"):
        with st.spinner("Updating index..."):
            st.session_state.rag.load_and_index("./documents")
        st.success("Refreshed!")

# Chat Window
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about requirements, authors, or test protocols..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = st.session_state.rag.get_answer(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})