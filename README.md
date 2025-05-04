# Bicycle Documentation System

This system uses LlamaIndex to create a searchable knowledge base from bicycle documentation.

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