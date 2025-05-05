# query_vectorstore.py (fixed)
import os

# 1. Disable Streamlit’s file-watcher path inspection entirely
#    (stops the “Examining the path of torch.classes” messages)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import argparse
import torch

# 2. Patch torch.classes so Streamlit’s watcher won’t invoke torch._classes.__getattr__
try:
    torch.classes.__path__ = []
except Exception:
    pass

import logging
import asyncio
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Tuple, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from functools import lru_cache
import threading

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure there’s always an event loop for any async operations
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Global variables for caching
_embedding_model = None
_llm_model = None
_llm_tokenizer = None
_vector_store = None
_cache_lock = threading.Lock()

def initialize_embedding_model() -> HuggingFaceEmbedding:
    """Initialize and return the embedding model."""
    logger.info("Initializing embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder="./embed_cache"
    )
    Settings.embed_model = embed_model
    return embed_model

def initialize_llm(use_gpu: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Initialize LLM with CPU by default and MPS only if --use-gpu is set."""
    model_name = "Qwen/Qwen1.5-0.5B-Chat"
    device = "mps" if use_gpu and torch.backends.mps.is_available() else "cpu"
    logger.info(f"Loading {model_name} on {device}")

    # Use float32 for CPU and MPS
    torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_vector_store(index_dir: str) -> any:
    """Load and return the vector store index."""
    logger.info("Loading vector store index...")
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)
    return index

def format_context(docs: List[str]) -> str:
    """Format retrieved documents into a context string."""
    return "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])

def generate_response(model, tokenizer, question: str, context: str) -> str:
    """Generate response with MPS memory constraints."""
    max_context_length = 512
    truncated_context = context[:max_context_length]

    messages = [
        {"role": "system", "content": "Answer using these specs:"},
        {"role": "user", "content": f"Context:\n{truncated_context}\n\nQuestion: {question}"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    ).to(model.device)

    if model.device.type == 'mps':
        torch.mps.empty_cache()

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=128,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    if model.device.type == 'mps':
        torch.mps.empty_cache()

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

def get_cached_embedding_model() -> HuggingFaceEmbedding:
    """Get cached embedding model."""
    global _embedding_model
    with _cache_lock:
        if _embedding_model is None:
            _embedding_model = initialize_embedding_model()
    return _embedding_model

def get_cached_llm(use_gpu: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Get cached LLM model."""
    global _llm_model, _llm_tokenizer
    with _cache_lock:
        if _llm_model is None or _llm_tokenizer is None:
            _llm_model, _llm_tokenizer = initialize_llm(use_gpu)
    return _llm_model, _llm_tokenizer

def get_cached_vector_store(index_dir: str) -> any:
    """Get cached vector store index."""
    global _vector_store
    with _cache_lock:
        if _vector_store is None:
            _vector_store = get_vector_store(index_dir)
    return _vector_store

def query_index(question: str, index_dir: str = "./storage", use_gpu: bool = False) -> Optional[str]:
    """Main function to process a query through the vector store and LLM."""
    try:
        embed_model = get_cached_embedding_model()
        model, tokenizer = get_cached_llm(use_gpu)
        index = get_cached_vector_store(index_dir)

        logger.info("Retrieving relevant documents...")
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(question)

        docs = [node.text for node in nodes]
        context = format_context(docs)

        logger.info("Generating response with LLM...")
        return generate_response(model, tokenizer, question, context)

    except FileNotFoundError:
        logger.error(f"Index directory '{index_dir}' not found")
        return None
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--index-dir", type=str, default="./storage")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU-optimized model if available")
    args = parser.parse_args()

    answer = query_index(args.question, args.index_dir, args.use_gpu)
    if answer:
        print(f"\nAnswer: {answer}\n")



