import logging
import re
from typing import Counter, List, Set
import numpy as np
from sentence_transformers import SentenceTransformer, util
from abstract import AbstractPreValidator

class DuplicatePromptValidator(AbstractPreValidator):
    def validate_data(self, data: List[dict]) -> bool:
        prompts = [item["prompt"] for item in data]
        duplicates = [item for item, count in Counter(prompts).items() if count > 1]

        if duplicates:
            logging.warning(f"Duplicate prompts found: {duplicates}")
            return False
        return True


class TrainOnTestValidator(AbstractPreValidator):
    def __init__(self, test_set: Set[str], similarity_threshold: float = 0.9):
        self.test_set = test_set
        self.similarity_threshold = similarity_threshold
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def validate_data(self, data: List[dict]) -> bool:
        for item in data:
            submission_texts = [
                item["prompt"],  # Check similarity against prompt
                item["chain_of_thought"],  # Check similarity against reasoning
                item["final_answer"]  # Check similarity against final answer
            ]

            # TODO: Can be improved for checking test-in-train between different datapoints.
            for submission_text in submission_texts:
                for test_entry in self.test_set:
                    similarity = util.pytorch_cos_sim(
                        self.similarity_model.encode(submission_text, convert_to_tensor=True),
                        self.similarity_model.encode(test_entry, convert_to_tensor=True)
                    ).item()

                    if similarity > self.similarity_threshold:
                        logging.warning(f"Submission too similar to test data: {submission_text} (Sim: {similarity:.2f})")
                        return False
        return True


class ReasoningQualityValidator(AbstractPreValidator):
    def validate_data(self, data: List[dict]) -> bool:
        for item in data:
            reasoning = item["chain_of_thought"].strip()

            if len(reasoning.split()) < 5:
                logging.warning(f"Reasoning too short: {reasoning}")
                return False

            if not re.search(r"[A-Za-z]", reasoning):
                logging.warning(f"Reasoning contains no meaningful words: {reasoning}")
                return False

        return True


class DataDiversityValidator(AbstractPreValidator):   
    def validate_data(self, data: List[dict]) -> bool:
        embeddings = [SentenceTransformer("all-MiniLM-L6-v2").encode(item["chain_of_thought"]) for item in data]
        similarities = np.mean(
            [util.pytorch_cos_sim(embeddings[i], embeddings[j]).item() 
             for i in range(len(embeddings)) 
             for j in range(i + 1, len(embeddings))]
        ) if len(embeddings) > 1 else 0

        if similarities > 0.85:
            logging.warning(f"Reasoning texts are too similar (Average similarity: {similarities:.2f})")
            return False
        return True


class FactualConsistencyValidator(AbstractPreValidator):
    """
    Validates that the chain-of-thought reasoning is factually consistent.
    Uses a knowledge base to check for factual errors or hallucinations.
    """
    
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize the validator with a knowledge base and confidence threshold.
        
        Args:
            confidence_threshold: Minimum confidence score required to consider a fact valid
        """
        self.confidence_threshold = confidence_threshold
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Expanded knowledge base with 50 common facts across various domains
        self.knowledge_base = {
            # Physics and Astronomy
            "sky blue": "The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere.",
            "gravity": "Gravity is a fundamental force that attracts objects with mass toward each other.",
            "speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "solar system": "Our solar system consists of the Sun, eight planets, dwarf planets, moons, asteroids, and comets.",
            "black hole": "A black hole is a region of spacetime where gravity is so strong that nothing can escape from it.",
            "big bang": "The Big Bang theory states that the universe began as a singularity about 13.8 billion years ago.",
            "seasons": "Seasons are caused by Earth's axial tilt as it orbits the Sun.",
            "northern lights": "The Northern Lights (Aurora Borealis) are caused by solar particles interacting with Earth's magnetic field.",
            
            # Chemistry
            "water boiling": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
            "water freezing": "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) at standard atmospheric pressure.",
            "periodic table": "The periodic table organizes chemical elements by atomic number, electron configuration, and chemical properties.",
            "atom structure": "Atoms consist of a nucleus containing protons and neutrons, surrounded by electrons.",
            "carbon dating": "Carbon dating determines the age of organic materials by measuring the decay of carbon-14 isotopes.",
            "acid base": "Acids donate hydrogen ions (H+) in solution, while bases accept hydrogen ions or donate hydroxide ions (OH-).",
            
            # Biology
            "photosynthesis": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
            "dna structure": "DNA has a double helix structure composed of nucleotides containing four bases: adenine, thymine, guanine, and cytosine.",
            "cell theory": "All living organisms are composed of cells, cells are the basic unit of life, and all cells come from pre-existing cells.",
            "evolution": "Evolution by natural selection is the process by which organisms change over time as a result of changes in heritable traits.",
            "immune system": "The immune system is a complex network of cells, tissues, and organs that defends the body against infections.",
            "human genome": "The human genome contains approximately 3 billion base pairs and around 20,000-25,000 protein-coding genes.",
            
            # Earth Science
            "earth rotation": "The Earth rotates on its axis once every 24 hours, causing day and night.",
            "earth revolution": "Earth completes one revolution around the Sun in approximately 365.25 days.",
            "plate tectonics": "Plate tectonics is the theory that Earth's outer shell is divided into plates that move over the mantle.",
            "water cycle": "The water cycle describes the continuous movement of water on, above, and below Earth's surface through evaporation, condensation, and precipitation.",
            "climate change": "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities like burning fossil fuels.",
            "ocean currents": "Ocean currents are continuous, directed movements of seawater generated by forces like wind, temperature, and salinity differences.",
            
            # Mathematics
            "pythagorean theorem": "In a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the other two sides.",
            "pi value": "Pi (π) is the ratio of a circle's circumference to its diameter, approximately equal to 3.14159.",
            "prime numbers": "A prime number is a natural number greater than 1 that is not a product of two smaller natural numbers.",
            "fibonacci sequence": "The Fibonacci sequence is a series where each number is the sum of the two preceding ones, usually starting with 0 and 1.",
            
            # Computer Science
            "algorithm": "An algorithm is a finite sequence of well-defined instructions to solve a specific problem or perform a computation.",
            "binary code": "Binary code uses a two-symbol system, typically 0 and 1, to represent text or computer processor instructions.",
            "internet": "The Internet is a global system of interconnected computer networks that use TCP/IP to communicate.",
            "artificial intelligence": "Artificial intelligence is the simulation of human intelligence in machines programmed to think and learn like humans.",
            
            # History
            "world war ii": "World War II was a global war that lasted from 1939 to 1945, involving most of the world's nations.",
            "industrial revolution": "The Industrial Revolution was a period of transition to new manufacturing processes in Europe and the United States from about 1760 to 1840.",
            "ancient egypt": "Ancient Egyptian civilization flourished along the Nile River from about 3100 BCE to 332 BCE.",
            "roman empire": "The Roman Empire was the post-Republican period of ancient Rome, with extensive territories around the Mediterranean Sea in Europe, Africa, and Asia.",
            
            # Geography
            "continents": "Earth has seven continents: Africa, Antarctica, Asia, Europe, North America, Australia, and South America.",
            "mount everest": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
            "amazon river": "The Amazon River is the largest river by discharge volume of water in the world and the second longest.",
            "great barrier reef": "The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia.",
            
            # Literature and Arts
            "shakespeare": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language.",
            "mona lisa": "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci, considered one of the most famous paintings in the world.",
            
            # Miscellaneous
            "human body temperature": "The average normal body temperature for humans is approximately 37 degrees Celsius (98.6 degrees Fahrenheit).",
            "sound speed": "The speed of sound in dry air at 20°C (68°F) is approximately 343 meters per second.",
            "coffee caffeine": "Coffee contains caffeine, a natural stimulant that affects the central nervous system and can temporarily ward off drowsiness.",
            "chocolate origin": "Chocolate is made from the seeds of the Theobroma cacao tree, which originated in the tropical rainforests of the Americas.",
            "diamond composition": "Diamonds are composed of carbon atoms arranged in a crystal structure called diamond cubic."
        }
        
        # TODO: Generate embeddings for knowledge base statements
        self.knowledge_embeddings = {}

    def validate_data(self, data: List[dict]) -> bool:
        """
        TODO: Implement this
        Validate that the chain-of-thought reasoning is factually consistent. Feel free to use the knowledge base, 
        or if you have any other creative ideas to check factual consistency, please use that!
        
        Args:
            data: List of prompt data to validate
            
        Returns:
            True if all data is factually consistent, False otherwise
        """
        pass
