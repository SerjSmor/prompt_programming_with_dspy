import dspy
from dspy import BootstrapFewShotWithRandomSearch

from chapter_3.dspy_structures import ClosedIntentSignature
from chapter_4.evaluate_dspy import validate_answer
from chapter_5.utils import create_examples_from_set

optimizer = BootstrapFewShotWithRandomSearch(
    metric=validate_answer,
    max_bootstrapped_demos=4,
    max_labeled_demos=0,
    num_threads=10,
    num_candidate_programs=5
)

intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
valset = create_examples_from_set('test', 50)
train_examples = create_examples_from_set('train', 4700)
intent_classifier_bootstrap_few_shot_with_random_search = optimizer.compile(intent_classifier_cot,
                                                                            trainset=train_examples,
                                                                          valset=valset)
intent_classifier_bootstrap_few_shot_with_random_search(**train_examples[99].inputs())
lm.inspect_history(n=1)
