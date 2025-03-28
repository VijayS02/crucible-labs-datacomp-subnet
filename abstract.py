from abc import abstractmethod
from typing import List, TypedDict

class PromptData(TypedDict):
    prompt: str
    chain_of_thought: str 
    final_answer: str


class AbstractPreValidator:
    @abstractmethod
    def validate_data(self, data: List[PromptData]) -> bool:
        """
        Given a list of prompt datas, this method should return True if the data is valid, False otherwise.

        Potential implementations may connect with huggingface to validate commit ids, check for duplicates, etc. 
        """
        raise NotImplementedError("Subclasses must implement this method")
    

class AbstractScorer:
    @abstractmethod
    def score(self, output, expected) -> float:
        """
        Given a generated output and an expected output, this method should return a score between 0 and 1.
        """
        raise NotImplementedError("Subclasses must implement this method")

class AbstractCrucibleModel:
    @abstractmethod
    def predict(self, tokenized_input) -> list:
        """
        Note: This method should attempt to be as deterministic as possible. Try to prevent randomness
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def fine_tune(self, inputs):
        """
        Given a list of inputs in plaintext where the CoT and final answer are combined, this method should fine-tune the model.
        """
        raise NotImplementedError("Subclasses must implement this method")