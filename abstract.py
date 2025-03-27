from abc import abstractmethod

class AbstractPreValidator:
    @abstractmethod
    def validate_data(self, data):
        raise NotImplementedError("Subclasses must implement this method")
    

class AbstractScorer:
    @abstractmethod
    def score(self, output, expected):
        raise NotImplementedError("Subclasses must implement this method")
    

class AbstractCrucibleTokenizer:
    @abstractmethod
    def encode_batch(self, text):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def decode_batch(self, text):
        raise NotImplementedError("Subclasses must implement this method")
    
    
class AbstractCrucibleModel:
    @abstractmethod
    def predict(self, tokenized_input):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def fine_tune(self, inputs):
        raise NotImplementedError("Subclasses must implement this method")