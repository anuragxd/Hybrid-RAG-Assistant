import os
import yaml
import json
from dotenv import load_dotenv

from core.llm import GroqEngine
from core.embeddings import EmbeddingEngine
from indexing.manager import IndexManager
from evaluation.generator import SyntheticDataGenerator
from utils.logger import get_logger

load_dotenv()
logger = get_logger("Eval_Prep")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    logger.info("1. Loading LLM and Embedding models...")
    llm = GroqEngine(model=config["llm"]["model"])
    embedder = EmbeddingEngine(model_name=config["models"]["embedding"])
    
    logger.info("2. Loading existing Knowledge Base...")
    index_manager = IndexManager(embedder, config["indexing"])
    
    # --- THIS IS THE CRITICAL NEW PART ---
    if index_manager.load("saved_index"):
        logger.info("Successfully loaded Knowledge Base from disk.")
    else:
        logger.error("No saved index found. Please run the Streamlit UI and scrape a topic first.")
        return
        
    all_chunks = list(index_manager.chunk_store.values())
    
    if not all_chunks:
        logger.error("Index loaded, but no chunks found inside it.")
        return

    logger.info(f"Found {len(all_chunks)} chunks in the knowledge base.")
    
    logger.info("3. Initializing Synthetic Data Generator...")
    generator = SyntheticDataGenerator(llm)
    
    # Generate 10 questions
    qa_pairs = generator.generate_qa_pairs(all_chunks, num_pairs=10)
    
    logger.info("4. Saving Test Dataset to disk...")
    # Ensure the evaluation directory exists
    os.makedirs("evaluation", exist_ok=True)
    output_path = "evaluation/testset.json"
    
    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=4)
        
    logger.info(f"✅ Successfully generated {len(qa_pairs)} QA pairs and saved to {output_path}")

if __name__ == "__main__":
    main()