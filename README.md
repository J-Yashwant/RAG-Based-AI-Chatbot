# RAG-Based-AI-Chatbot

A Retrieval-Augmented Generation (RAG) system combines data retrieval with a language model to answer questions accurately using external information. Instead of relying only on the model’s memory, the system first reads and indexes source data such as CSV files, PDFs, text, or HTML. When a user asks a question, the system analyzes the query and decides how to respond. If the question involves calculations like totals, averages, or counts, the data is processed directly using tools such as Pandas to produce exact results. For descriptive or analytical questions, the system retrieves the most relevant data using a hybrid approach that blends keyword-based search (BM25) with semantic search through embeddings. The retrieved information is then passed to a language model, which generates an answer strictly based on the provided context. This approach ensures accurate, up-to-date, and explainable responses while preventing hallucinations and making the system adaptable to different types of datasets and domains.


Environment Setup:

step-1: Install Ollama
step-2: Open the terminal
step-3: Run the Command "ollama pull qwen2.5:7b"
step-4: After installation start ollama just by using the windows explorer and open it
step-5: Now the Ollama environment it set
step-6: Install all the packages in the requirements file
step-7: Run the File 
