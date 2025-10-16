from typing import Literal

import dspy
import pydantic
from pydantic import Field

# Define the model for a single contact

class Answer(pydantic.BaseModel):
    reasoning: str = Field(description="Reasoning under 10 words")
    answer: str
    confidence: Literal['HIGH', 'MEDIUM', 'LOW']


class AnswerSignature(dspy.Signature):
    question: str = dspy.InputField()
    answer: Answer = dspy.OutputField()


lm = dspy.LM("gpt-4o-mini") # or any other supported LM
dspy.configure(lm=lm)

prog = dspy.Predict(AnswerSignature)
prediction = prog(question="What is the capital of France?")
print(prediction)