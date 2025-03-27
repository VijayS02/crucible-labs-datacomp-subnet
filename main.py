import sys
from typing import List, TypedDict

from abstract import AbstractPreValidator, AbstractScorer, AbstractCrucibleTokenizer, AbstractCrucibleModel 

class PromptData(TypedDict):
    task: str
    chain_of_thought: str 
    final_answer: str


class Validator:
    def __init__(self, pre_validators: List[AbstractPreValidator], scorers: List[AbstractScorer]):
        self.pre_validators = pre_validators
        self.scorers = scorers


    def forward_pass(self, model, tokenizer, data: List[PromptData]):
        tokenized_outputs = []
        inputs = [tokenizer.encode_batch(item['prompt']) for item in data]
        for tokenized_input in inputs: 
            output = model.predict(tokenized_input)
            tokenized_outputs.append(output[0])
        
        decoded_outputs = tokenizer.decode_batch(tokenized_outputs)
        return decoded_outputs
    
    def prompt_combine(self, prompt: PromptData):
        return f"{prompt['task']}\nReasoning: {prompt['chain_of_thought']}\nAnswer:  {prompt['final_answer']}"  

    
    def fine_tune(self, model: AbstractCrucibleModel, tokenizer: AbstractCrucibleTokenizer, data: List[PromptData]):
        texts = [item['chain_of_thought'] + "\n\nAnswer" for item in data]
        inputs = tokenizer.encode_batch(texts)
        model.fine_tune(inputs)


    def pre_validate(self, data: List[PromptData]):
        for pre_validator in self.pre_validators:
            if not pre_validator.validate_data(data):
                print(f"Data is not valid for {pre_validator.__class__.__name__}", file=sys.stderr)
                return False
        return True

    def validate_and_score(self, model: AbstractCrucibleModel, tokenizer: AbstractCrucibleTokenizer, data: List[PromptData], tune=False):
        if tune:
            self.fine_tune(model, tokenizer, data)

        output = self.forward_pass(model, tokenizer, data)


        similarities = []
        expected_results = [item["final_answer"] for item in data ]

        for output, expected in zip(output, expected_results):
            sim_score = 0
            for scorer in self.scorers:
                sim_score += scorer.score(output, expected)
            avg_sim_score = sim_score / len(self.scorers)
            similarities.append(avg_sim_score)

        if len(similarities) == 0:
            return 0

        return sum(similarities) / len(similarities) 
    
    def test(self, model: AbstractCrucibleModel, tokenizer: AbstractCrucibleTokenizer, data: List[PromptData]):
        if not self.pre_validate(data):
            raise ValueError("Data is not valid")

        original_score = self.validate_and_score(model, tokenizer, data)

        trained_score = self.validate_and_score(model, tokenizer, data, tune=True)

        print(f"Original score: {original_score}")

        print(f"Trained score: {trained_score}")

        return trained_score
        
