from typing import List
from abstract import AbstractCrucibleModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class PytorchModelHF(AbstractCrucibleModel):
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    def predict(self, raw_input: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(raw_input, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_length=100)
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def fine_tune(self, raw_inputs: List[str], batch_size: int = 2, epochs: int = 1):
        # self.model.train()
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # inputs = self.tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
        # dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # for epoch in range(epochs):
        #     total_loss = 0
        #     for batch in dataloader:
        #         input_ids, attention_mask = batch
        #         optimizer.zero_grad()

        #         outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        #         loss = outputs.loss
        #         loss.backward()

        #         optimizer.step()
        #         total_loss += loss.item()

        #     print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")
        pass 

if __name__ == "__main__":
    model = PytorchModelHF("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    text = "Explain why the sky is blue."
    output = model.predict(text)
    print("Generated Output:", output)