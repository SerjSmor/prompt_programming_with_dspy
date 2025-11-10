import dspy

from chapter_3.dspy_structures import ClosedIntentSignature
from chapter_4.evaluate_dspy import validate_answer
from chapter_5.utils import create_examples_from_set

small_train_set = create_examples_from_set('train', 50)
train_examples = create_examples_from_set('train', 100)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)

copro = dspy.COPRO(metric=validate_answer, breadth=2, depth=2, track_stats=True)
intent_classifier = dspy.Predict(ClosedIntentSignature)
optimized_intent_classifier = copro.compile(intent_classifier, trainset=small_train_set, eval_kwargs={"num_threads": 3})

print(optimized_intent_classifier.results_best)
# Output: {140705299144544: {'depth': [0, 1], 'max': [90.0, 100.0], 'average': [90.0, 95.0], 'min': [90.0, 90.0], 'std': [np.float64(0.0), np.float64(5.0)]}}
print(optimized_intent_classifier.results_latest)
# Output: {140705299144544: {'depth': [0, 1], 'max': [90.0, 100.0], 'average': [90.0, 95.0], 'min': [90.0, 90.0], 'std': [np.float64(0.0), np.float64(5.0)]}}
print(optimized_intent_classifier.total_calls)
# Output: 4

optimized_intent_classifier(**train_examples[99].with_inputs())
lm.inspect_history(n=1)
