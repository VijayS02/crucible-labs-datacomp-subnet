from abstract import AbstractScorer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


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
    
class RougeScorer(AbstractScorer):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def score(self, output: str, expected: str) -> float:
        if not output or not expected:
            return 0.0

        return self.scorer.score(output, expected)["rougeL"].fmeasure
    
class BleuScorer(AbstractScorer):
    def score(self, output: str, expected: str) -> float:
        """Computes BLEU score for word overlap precision."""
        if not output or not expected:
            return 0.0

        output_tokens = output.lower().split()
        expected_tokens = expected.lower().split()

        smoother = SmoothingFunction().method1

        return sentence_bleu([expected_tokens], output_tokens, smoothing_function=smoother)

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
