import unittest
from prevalidators import FactualConsistencyValidator

class TestFactualConsistencyValidator(unittest.TestCase):
    def setUp(self):
        self.validator = FactualConsistencyValidator(confidence_threshold=0.7)
    
    def test_valid_data(self):
        valid_data = [
            {
                "prompt": "Explain why the sky is blue:",
                "chain_of_thought": "The sky appears blue because of Rayleigh scattering. Sunlight consists of all colors, but when it enters our atmosphere, the shorter blue wavelengths scatter more than other colors.",
                "final_answer": "The sky appears blue because molecules in the atmosphere scatter shorter wavelengths (blue light) more than longer wavelengths."
            }
        ]
        self.assertTrue(self.validator.validate_data(valid_data))
    
    def test_invalid_data(self):
        invalid_data = [
            {
                "prompt": "Explain why the sky is blue:",
                "chain_of_thought": "The sky is blue because it reflects the color of the oceans. The water on Earth makes the sky look blue through a process of reflection.",
                "final_answer": "The sky is blue because it reflects the blue color of the oceans."
            }
        ]
        self.assertFalse(self.validator.validate_data(invalid_data))

    # TODO: Add tests for the knowledge base and correctness

if __name__ == '__main__':
    unittest.main() 