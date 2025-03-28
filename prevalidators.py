import logging
import re
from typing import Counter, List, Set
import numpy as np
from sentence_transformers import SentenceTransformer, util
from abstract import AbstractPreValidator

class DuplicatePromptValidator(AbstractPreValidator):
    def validate_data(self, data: List[dict]) -> bool:
        prompts = [item["prompt"] for item in data]
        duplicates = [item for item, count in Counter(prompts).items() if count > 1]

        if duplicates:
            logging.warning(f"Duplicate prompts found: {duplicates}")
            return False
        return True


class TrainOnTestValidator(AbstractPreValidator):
    def __init__(self, test_set: Set[str], similarity_threshold: float = 0.9):
        self.test_set = test_set
        self.similarity_threshold = similarity_threshold
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def validate_data(self, data: List[dict]) -> bool:
        for item in data:
            submission_texts = [
                item["prompt"],  # Check similarity against prompt
                item["chain_of_thought"],  # Check similarity against reasoning
                item["final_answer"]  # Check similarity against final answer
            ]

            for submission_text in submission_texts:
                for test_entry in self.test_set:
                    similarity = util.pytorch_cos_sim(
                        self.similarity_model.encode(submission_text, convert_to_tensor=True),
                        self.similarity_model.encode(test_entry, convert_to_tensor=True)
                    ).item()

                    if similarity > self.similarity_threshold:
                        logging.warning(f"Submission too similar to test data: {submission_text} (Sim: {similarity:.2f})")
                        return False
        return True


class ReasoningQualityValidator(AbstractPreValidator):
    def validate_data(self, data: List[dict]) -> bool:
        for item in data:
            reasoning = item["chain_of_thought"].strip()

            if len(reasoning.split()) < 5:
                logging.warning(f"Reasoning too short: {reasoning}")
                return False

            if not re.search(r"[A-Za-z]", reasoning):
                logging.warning(f"Reasoning contains no meaningful words: {reasoning}")
                return False

        return True


class DataDiversityValidator(AbstractPreValidator):   
    def validate_data(self, data: List[dict]) -> bool:
        embeddings = [SentenceTransformer("all-MiniLM-L6-v2").encode(item["chain_of_thought"]) for item in data]
        similarities = np.mean(
            [util.pytorch_cos_sim(embeddings[i], embeddings[j]).item() 
             for i in range(len(embeddings)) 
             for j in range(i + 1, len(embeddings))]
        ) if len(embeddings) > 1 else 0

        if similarities > 0.85:
            logging.warning(f"Reasoning texts are too similar (Average similarity: {similarities:.2f})")
            return False
        return True