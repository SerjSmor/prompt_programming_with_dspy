import dspy

from chapter_3.dspy_structures import ClosedIntentSignature
from chapter_4.evaluate_dspy import validate_answer
from chapter_5.utils import evaluate_model, create_examples_from_set

small_train_set = create_examples_from_set('train', 150)
dev_set = create_examples_from_set('test', 150)

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)
model = dspy.Predict(ClosedIntentSignature)
mipro = dspy.MIPROv2(metric=validate_answer, auto="light")
optimized_model = mipro.compile(model, trainset=small_train_set, max_bootstrapped_demos=3,
                                max_labeled_demos=4, valset=dev_set,
                                requires_permission_to_run=False)

optimized_model(message="How can I list my flight plan?")
lm.inspect_history()
# evaluate_model(val_examples, lm, optimized_model)

