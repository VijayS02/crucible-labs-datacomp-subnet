from abstract import AbstractScorer


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
