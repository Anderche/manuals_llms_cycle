# RAG LLM App for Contextual Document Retrieval


## Description

This system uses LlamaIndex to create a searchable knowledge base from bicycle documentation. This project is a document query system built with Python, leveraging LlamaIndex and local AI models for privacy-focused document processing. The tech stack includes **_LlamaParse_** for **PDF parsing**, **_BAAI/bge-small-en-v1.5_** for **embeddings**, and **_Qwen/Qwen1.5-0.5B-Chat_** that serves as the **large language model (LLM) for natural language processing querying**. 

More specifically, the embeddings model (BAAI/bge-small-en-v1.5) converts text into numerical vectors for semantic searching, whereas the LLM (Qwen/Qwen1.5-0.5B-Chat) generates natural language responses based on retrieved context. The system processes .pdf documents hierarchically, creating a vector store for efficient semantic search. 

Most importantly, this system designed to be extensible to any business domain - beyond cycling documents - by simply replacing the input documents; the core architecture remains the same. Businesses can use it to create a searchable knowledge base from their internal documentation, enabling natural language queries about company policies, product specifications, or technical documentation. The system runs locally, ensuring data privacy and security while providing accurate, context-aware responses to user queries.

--- 

## Business Use-Cases

1. This app can serve as an **internal knowledge base for HR policies, product documentation, and technical manuals**.

2. It is ideal for **customer support teams** needing quick access to product specs, legal departments searching through contracts, and **training departments** creating interactive learning materials. 

3. The system can also help with **compliance documentation** and **standard operating procedures** across industries.

--- 

## Files

- `parse_to_vectorstore.py`: Processes PDF documents into a vector store using LlamaParse and BAAI/bge-small-en-v1.5 embeddings. Splits content hierarchically for better context retention.

- `query_vectorstore.py`: Queries the vector store using Qwen/Qwen1.5-0.5B-Chat for natural language responses. Uses local embeddings for efficient similarity search. The script uses a local embedding model (BAAI/bge-small-en-v1.5) to process text and a vector store (stored in the specified directory) to retrieve relevant information. Once the model downloads complete, it will use this setup to answer your question about Bianchi's carbon frames.

## Reference

Any .pdf document (containing a mix of text and tabular information) can serve as input for this system's internal _knowledge-base_. For this application, a partial reference .pdf (preserving only pages containing cycle specifications, not graphics) was utilized as the _knowledge-base_. The reference .pdf (Bianchi's 2022 cycle catalog) can be found here:

    - https://www.bianchi.com/wp-content/uploads/2022/02/202202_BianchiTechnicalBook-1.pdf


## Setup

1. Create and activate a virtual environment (choose one method):

   Using conda:
   ```bash
   conda create -n bicycle-docs python=3.9
   conda activate bicycle-docs
   ```

   Or using Python's venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Process PDFs:
```bash
python parse_to_vectorstore.py --pdf your_document.pdf
```

4. Query the knowledge base:
```bash
python query_vectorstore.py --question "Your question here"
```

The system uses local models for privacy and efficiency, with BAAI/bge-small-en-v1.5 for embeddings and Qwen/Qwen1.5-0.5B-Chat for question answering. 

