from abc import abstractmethod
from typing import List, TypedDict

class PromptData(TypedDict):
    prompt: str
    chain_of_thought: str 
    final_answer: str


class AbstractPreValidator:
    @abstractmethod
    def validate_data(self, data: List[PromptData]) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
    

class AbstractScorer:
    @abstractmethod
    def score(self, output, expected) -> float:
        raise NotImplementedError("Subclasses must implement this method")

class AbstractCrucibleModel:
    @abstractmethod
    def predict(self, tokenized_input) -> list:
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def fine_tune(self, inputs):
        raise NotImplementedError("Subclasses must implement this method")