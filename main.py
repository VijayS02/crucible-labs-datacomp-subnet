import sys
from typing import List

from abstract import AbstractPreValidator, AbstractScorer, AbstractCrucibleModel, PromptData
from models import PytorchModelHF
from prevalidators import DuplicatePromptValidator, TrainOnTestValidator, ReasoningQualityValidator, DataDiversityValidator
from scorers import SemanticScorer 
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Validator:
    def __init__(self, pre_validators: List[AbstractPreValidator], scorers: List[AbstractScorer]):
        self.pre_validators = pre_validators
        self.scorers = scorers


    def forward_pass(self, model: AbstractCrucibleModel, data: List[PromptData]):
        """
        Runs forward passes on the model and returns the outputs.

        Note: This method should try to be as deterministic as possible. 
        """
        outputs = []
        inputs = [item['prompt'] for item in data]
        for input in inputs: 
            output = model.predict(input)
            outputs.append(output)

        return outputs
    
    def prompt_combine(self, prompt: PromptData):
        """
        For a single prompt, this function will combine the prompt, chain of thought, and final answer into a single string.
        """
        return f"{prompt['prompt']}\nReasoning: {prompt['chain_of_thought']}\nAnswer:  {prompt['final_answer']}"  

    
    def fine_tune(self, model: AbstractCrucibleModel, data: List[PromptData]):
        """
        Given the miner's data, this will run promt_combine on each item fine tune the model.
        """
        texts = [self.prompt_combine(item) for item in data]
        model.fine_tune(texts)


    def pre_validate(self, data: List[PromptData]):
        """
        Given a list of prompt data, this method will run all prevalidators and return True if the data is valid, False otherwise.
        """
        for pre_validator in self.pre_validators:
            if not pre_validator.validate_data(data):
                print(f"Data is not valid for {pre_validator.__class__.__name__}", file=sys.stderr)
                return False
        return True

    def validate_and_score(self, model: AbstractCrucibleModel, data: List[PromptData], tune=False) -> float:
        """
        As described in the specification, this will fine tune if necessary, run a forward pass, and compute similarity scores. 
        """
        if tune:
            logging.info("Fine tuning enabled - running fine tuning")
            self.fine_tune(model, data)

        logging.info("Running forward pass")
        output = self.forward_pass(model, data)

        logging.info("Computing scores")
        similarities = []
        expected_results = [item["final_answer"] for item in data ]

        for output, expected in zip(output, expected_results):
            sim_score = 0
            for scorer in self.scorers:
                sim_score += scorer.score(output, expected)
            avg_sim_score = sim_score / len(self.scorers)
            logging.debug(f"Similarity score: {avg_sim_score}")
            similarities.append(avg_sim_score)

        if len(similarities) == 0:
            logging.warning("No similarities found")
            return 0

        logging.debug(f"Similarities: {similarities}")
        return sum(similarities) / len(similarities) 
    
    def test(self, model: AbstractCrucibleModel, data: List[PromptData]) -> float:
        """
        Given a model and a list of prompt data, this method will run a single test/validation run. 

        Note: In order to counteract randomness, the test function can run multiple experiments and return an average ratio. 

        :return: The ratio of the trained score to the original score.
        """
        if not self.pre_validate(data):
            raise ValueError("Data is not valid")

        original_score = self.validate_and_score(model, data)

        trained_score = self.validate_and_score(model, data, tune=True)

        print(f"Original score: {original_score}")

        print(f"Trained score: {trained_score}")

        return trained_score/original_score
        


if __name__ == "__main__":

    test_set = {
        "The sky appears blue due to Rayleigh scattering of sunlight.",
        "Paris is the capital of France."
    }

    pre_validators = [
        DuplicatePromptValidator(),
        TrainOnTestValidator(test_set),
        ReasoningQualityValidator(),
        DataDiversityValidator(),
    ]

    validator = Validator(pre_validators, [SemanticScorer()])

    model = PytorchModelHF("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")



    bad_data = [
        {
            "prompt": "Explain why the sky is blue.",
            "chain_of_thought": "The sky is blue because of Rayleigh scattering.",
            "final_answer": "The sky is blue because of Rayleigh scattering."
        },
        {
            "prompt": "What is the capital of France?",
            "chain_of_thought": "The capital of France is Paris.",
            "final_answer": "The capital of France is Paris."
        }
    ]

    good_data = [
        {
            "prompt": "Explain why the sky is blue:",
            "chain_of_thought": "Consider Rayleigh scattering and how molecules scatter short-wavelength light.",
            "final_answer": "The sky appears blue because molecules in the atmosphere scatter shorter wavelengths more strongly."
        },
        {
            "prompt": "What is 2 + 2?",
            "chain_of_thought": "Straightforward math. 2 + 2 = 4",
            "final_answer": "4"
        }
    ]

    validator.test(model, good_data)