import dspy
from dspy import BootstrapFewShot

from chapter_3.dspy_structures import ClosedIntentSignature
from chapter_4.evaluate_dspy import validate_answer
from chapter_5.utils import create_examples_from_set

optimizer = BootstrapFewShot(
    validate_answer,
    max_bootstrapped_demos=4,
    max_labeled_demos=6,
    max_rounds=10
)

intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
train_examples = create_examples_from_set("train", 4700)
bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
bootrstrap_few_shot(**train_examples[99].inputs())
lm.inspect_history(n=1)
