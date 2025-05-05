from typing import List, Dict, Any
from data_processing.parse_to_vectorstore import parse_document
from data_processing.query_vectorstore import query_documents

class RAGPipeline:
    def __init__(self):
        self.vectorstore = None
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize the RAG pipeline components."""
        pass
    
    def process_document(self, document_path: str):
        """Process a document and add it to the vectorstore."""
        pass
    
    def query(self, query: str) -> Dict[str, Any]:
        """Query the RAG pipeline with a user question."""
        pass
