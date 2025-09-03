import dspy
from typing import List


class IntentSignature(dspy.Signature):
    """
    Classify the message into one of the possible labels.
    """
    message: str = dspy.InputField()
    labels: List[str] = dspy.InputField()
    intent_label: str = dspy.OutputField()
