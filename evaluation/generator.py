import json
import random
from typing import List, Dict
from core.llm import GroqEngine
from utils.schemas import Chunk
from utils.logger import get_logger

logger = get_logger("Testset_Generator")

class SyntheticDataGenerator:
    def __init__(self, llm: GroqEngine):
        self.llm = llm

    def generate_qa_pairs(self, chunks: List[Chunk], num_pairs: int = 10) -> List[Dict]:
        qa_dataset = []
        
        # Randomly sample chunks to ensure diverse questions across the topic
        if len(chunks) > num_pairs:
            sampled_chunks = random.sample(chunks, num_pairs)
        else:
            sampled_chunks = chunks

        logger.info(f"Generating {len(sampled_chunks)} synthetic Q&A pairs...")

        prompt_template = """
        You are an expert educational evaluator. Given the following text context, your task is to generate one complex, specific question that can be answered SOLELY using this context. 
        Then, provide the comprehensive and factually correct answer based on the context.

        Context:
        {context_text}

        Respond STRICTLY in the following JSON format, with no additional text, pleasantries, or markdown formatting outside the JSON:
        {{
            "question": "Your generated question here",
            "ground_truth": "The factual answer based on the context"
        }}
        """

        for i, chunk in enumerate(sampled_chunks):
            prompt = prompt_template.format(context_text=chunk.text)
            # Use a very low temperature for deterministic, factual extraction
            response = self.llm.generate(prompt, system_prompt="You are a strict JSON-generating evaluation engine.")
            
            try:
                # Clean the response in case the LLM wrapped it in markdown (```json ... ```)
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:-3].strip()
                elif cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[3:-3].strip()
                    
                data = json.loads(cleaned_response)
                
                qa_dataset.append({
                    "question": data.get("question", ""),
                    "ground_truth": data.get("ground_truth", ""),
                    "reference_context": [chunk.text], # RAGAS expects a list of context strings
                    "source_url": chunk.metadata.url
                })
                logger.info(f"Generated QA pair {i+1}/{len(sampled_chunks)}")
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON for a chunk. LLM output was irregular. Skipping.")
                continue

        return qa_dataset