# pip install llama-index llama-parse python-dotenv
import argparse
import logging
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.WARNING)

load_dotenv()

def process_pdf(pdf_path: str, index_dir: str = "./storage"):
    try:
        logger.info("Initializing PDF parser...")
        parser = LlamaParse(
            result_type="markdown",
            disable_image_extraction=True,
            verbose=True  # Enable LlamaParse internal logging
        )
        
        logger.info(f"Processing PDF: {Path(pdf_path).name}")
        documents = SimpleDirectoryReader(
            input_files=[pdf_path],
            file_extractor={".pdf": parser}
        ).load_data()
        logger.info(f"Loaded {len(documents)} document sections")
        
        logger.info("Chunking document hierarchy...")
        from llama_index.core.node_parser import HierarchicalNodeParser
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        
        logger.info("Setting up local HuggingFace embedding model...")
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        logger.info("Creating vector store index with local embeddings...")
        index = VectorStoreIndex.from_documents(
            documents, 
            transformations=[node_parser],
            embed_model=embed_model
        )
        
        logger.info(f"Persisting index to {index_dir}")
        Path(index_dir).mkdir(exist_ok=True)
        index.storage_context.persist(persist_dir=index_dir)
        logger.info("Indexing completed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
        parser.add_argument("--index-dir", type=str, default="./storage", help="Output directory")
        args = parser.parse_args()
        
        logger.info(f"Starting processing for {args.pdf}")
        logger.info(f"Output directory: {args.index_dir}")
        
        process_pdf(args.pdf, args.index_dir)
        
    except SystemExit:
        logger.error("Invalid command line arguments")
        raise
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
        exit(1)


