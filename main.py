import sys 

class Validator:
    def __init__(self, pre_validators, scorers):
        self.pre_validators = pre_validators
        self.scorers = scorers


    def forward_pass(self, model, tokenizer, data):
        outputs = []
        for item in data:
            prompt = item['prompt']
            inputs = tokenizer.encode(prompt, data)
            output = model.predict(inputs)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            outputs.append(decoded)
        return outputs
    
    def prompt_combine(self, prompt):
        return f"{prompt['task']}\nReasoning: {prompt['chain_of_thought']}\nAnswer:  {prompt['final_answer']}"  

    
    def fine_tune(self, model, tokenizer, data):
        texts = [item['chain_of_thought'] + "\n\nAnswer" for item in data]
        inputs = tokenizer.encode(texts)
        model.fine_tune(inputs)


    def pre_validate(self, data):
        for pre_validator in self.pre_validators:
            if not pre_validator.validate_data(data):
                print(f"Data is not valid for {pre_validator.__class__.__name__}", file=sys.stderr)
                return False
        return True

    def validate_and_score(self, model, tokenizer, data, tune=False):
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
    
    def test(self, model, tokenizer, data):
        if not self.pre_validate(data):
            raise ValueError("Data is not valid")

        original_score = self.validate_and_score(model, tokenizer, data)

        trained_score = self.validate_and_score(model, tokenizer, data, tune=True)

        print(f"Original score: {original_score}")

        print(f"Trained score: {trained_score}")

        return trained_score
        
