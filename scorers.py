from abstract import AbstractScorer
from sentence_transformers import SentenceTransformer, util

class SemanticScorer(AbstractScorer):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, output: str, expected: str) -> float:
        """
        Computes semantic similarity between generated and expected outputs.
        """
        output_embedding = self.model.encode(output, convert_to_tensor=True)
        expected_embedding = self.model.encode(expected, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(output_embedding, expected_embedding).item())

class SimpleOverlapScorer(AbstractScorer):
    def score(self, output: str, expected: str) -> float:
        """
        Computes a simple word overlap score:
        - (Number of matching words) / (Number of words in the expected answer)
        """
        output_words = set(output.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 0.0  # Avoid division by zero

        overlap = len(output_words.intersection(expected_words))
        return overlap / len(expected_words)
