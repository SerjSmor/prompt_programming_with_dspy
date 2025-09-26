import dspy
from dspy import LabeledFewShot
from chapter_3.dspy_structures import IntentSignature
from chapter_5.utils import create_examples_from_set

train_examples = create_examples_from_set("train", 4700)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
num_examples = 4
intent_classifier_cot = dspy.ChainOfThought(IntentSignature)
labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
prediction = few_shot_model(**train_examples[99].inputs())
lm.inspect_history(n=1)
