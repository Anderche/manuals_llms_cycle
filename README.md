# Bicycle Documentation System

This system uses LlamaIndex to create a searchable knowledge base from bicycle documentation. This project is a document query system built with Python, leveraging LlamaIndex and local AI models for privacy-focused document processing. The tech stack includes **LlamaParse** for PDF parsing, BAAI/bge-small-en-v1.5 for **embeddings**, and the **LLM**, Zephyr-7B, for natural language processing querying. 

More specifically, the embeddings model (BAAI/bge-small-en-v1.5) converts text into numerical vectors for semantic searching, whereas the LLM (Zephyr-7B) generates natural language responses based on retrieved context.

The system processes .pdf documents hierarchically, creating a vector store for efficient semantic search. 

Most importantly, this system designed to be extensible to any business domain - beyond cycling documents - by simply replacing the input documents; the core architecture remains the same. Businesses can use it to create a searchable knowledge base from their internal documentation, enabling natural language queries about company policies, product specifications, or technical documentation. The system runs locally, ensuring data privacy and security while providing accurate, context-aware responses to user queries.



## Files

- `parse_to_vectorstore.py`: Processes PDF documents into a vector store using LlamaParse and BAAI/bge-small-en-v1.5 embeddings. Splits content hierarchically for better context retention.

- `query_vectorstore.py`: Queries the vector store using Zephyr-7B for natural language responses. Uses local embeddings for efficient similarity search. It currently uses the Zephyr-7B language model (about 15GB total) which will be used to answer questions about bicycle specifications. The script uses a local embedding model (BAAI/bge-small-en-v1.5) to process text and a vector store (stored in the specified directory) to retrieve relevant information. Once the model downloads complete, it will use this setup to answer your question about Bianchi's carbon frames.

## Reference

The .pdf for Bianchi's 2019 bicycles can be found here:

    - https://www.bianchi.com/wp-content/uploads/2019/12/Technical-Book-International-2019.pdf

## Setup

1. Install dependencies:
```bash
pip install llama-index llama-parse python-dotenv
```

2. Process PDFs:
```bash
python parse_to_vectorstore.py --pdf your_document.pdf
```

3. Query the knowledge base:
```bash
python query_vectorstore.py --question "Your question here"
```

The system uses local models for privacy and efficiency, with BAAI/bge-small-en-v1.5 for embeddings and Zephyr-7B for question answering. 