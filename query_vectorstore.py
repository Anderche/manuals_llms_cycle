# query_vectorstore.py
import argparse
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

def query_index(question: str, index_dir: str = "./storage"):
    # Configure local models FIRST
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    Settings.llm = HuggingFaceLLM(
        model_name="HuggingFaceH4/zephyr-7b-beta",
        tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
        context_window=3900,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.1}
    )
    
    # Load index
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)
    
    # Query
    query_engine = index.as_query_engine()
    print(f"Answer: {query_engine.query(question)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--index-dir", type=str, default="./storage")
    args = parser.parse_args()
    
    query_index(args.question, args.index_dir)



