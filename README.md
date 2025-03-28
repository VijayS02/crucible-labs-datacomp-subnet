# Datacomp Subnet Technical Interview

This is my submission for the crucible labs datacomp technical interview.

## TLDR: How do I run

I would recommend setting up a venv then installing requirements:

```
python -m venv venv
venv\Scripts\activate
```

Then install requirements:

```
pip install -r requirements.txt
```

Finally, run main:

```
python3 main.py
```

## Implementation details

In order to make my code extensible, I've created a few abstract classes which allow tests to easily be made and validators to run easily.
This allows easy integration of:

- New models
- New validators
- Custom scoring functions

### Abstract Base Classes

| Class                 | Purpose                                                                         |
| --------------------- | ------------------------------------------------------------------------------- |
| AbstractCrucibleModel | Provides a standard interface for models (e.g., predict(), fine_tune()).        |
| AbstractPreValidator  | Ensures that data submissions meet quality & security standards.                |
| AbstractScorer        | Defines a standard interface for evaluating model outputs against ground truth. |

### Implementations of abstract classes

To get started, I've created some basic implementations of the above. In [models.py](./models.py), I've used a basic `pytorch` implementation to load and run models from huggingface.

In [prevalidators.py](./prevalidators.py), I implemented a few pre-validators that use lightweight `sentence_transformers` to do things like duplicate, test-on-train, reasoning quality, and data diversity detection.

Similary, I used `sentence_transformers` in my [scorers.py](./scorers.py) to do basic semantic similarity scoring.

### Security & Exploit detection

To prevent miners from cheating or submitting low quality data, I use multiple security checks before accepting their submissions. First, I block duplicate data using a DuplicatePromptValidator, which detects repeated prompts using basic text equality, and a DataDiversityValidator, which checks for paraphrased copies with AI checking. To stop miners from submitting evaluation (test) data as training data, the TrainOnTestValidator compares new submissions against a known test set. If a miner’s answer is too similar to test data, it gets rejected.

To ensure quality reasoning, I use a ReasoningQualityValidator that enforces a minimum word count and that there are actually words. This prevents miners from submitting short or meaningless explanations.

By no means are these current implementations effective. To improve, I could use more advanced, AI-powered models to help do better checks, additionally, as provided in the datacomp subnet, various methods could be further implemented to improve the pre-validation. E.g. Tulu3, huggingface commit checks, limit dataset size, etc...

There is plenty of room for improvement in the pre-validation and scoring methodologies.
