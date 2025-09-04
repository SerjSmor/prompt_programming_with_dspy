import dspy
from dotenv import load_dotenv
from dspy import Evaluate

from consts import GPT_4O_MINI
from utils import create_examples_from_set, validate_answer, evaluate_model
from chapter_2_clasifying_users_intent.dspy_structure import IntentClassifier
load_dotenv()

lm = dspy.LM(GPT_4O_MINI, cache=False)
dspy.settings.configure(lm=lm)

val_examples = create_examples_from_set('test', 839)
small_train_set = create_examples_from_set('train', 50)
medium_train_set = create_examples_from_set('train', 200)
tiny_train_set = create_examples_from_set('train', 10)


def copro_intent_classifier():
    copro = dspy.COPRO(metric=validate_answer, breadth=4, depth=4, track_stats=True)
    intent_classifier = dspy.Predict(IntentClassifier)
    print(intent_classifier.signature)
    print()
    intent_classifier(**val_examples[99].with_inputs())
    lm.inspect_history(n=1)
    optimized_intent_classifier = copro.compile(intent_classifier, trainset=small_train_set,
                                                eval_kwargs={"num_threads": 3})
    print(optimized_intent_classifier.results_best)
    print(optimized_intent_classifier.results_latest)
    print(optimized_intent_classifier.total_calls)
    optimized_intent_classifier(**val_examples[99].with_inputs())
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, optimized_intent_classifier)

def copro_cot_intent_classifier():
    cot_intent_classifier = dspy.ChainOfThought(IntentClassifier)
    copro = dspy.COPRO(metric=validate_answer, breadth=4, depth=4, track_stats=True)
    optimized_cot_intent_classifier = copro.compile(cot_intent_classifier, trainset=small_train_set,
                                                eval_kwargs={"num_threads": 3})
    print(optimized_cot_intent_classifier.results_best)
    print(optimized_cot_intent_classifier.results_latest)
    print(optimized_cot_intent_classifier.total_calls)
    optimized_cot_intent_classifier(**val_examples[99].with_inputs())
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, optimized_cot_intent_classifier)


# def mipro_intent_classifier(base_model: dspy.Module):
#     print(f"Base model: {base_model}")
#     mipro = dspy.MIPROv2(metric=validate_answer, auto="light")
#     optimized_model = mipro.compile(base_model, trainset=small_train_set, max_bootstrapped_demos=3, max_labeled_demos=4,
#                                     requires_permission_to_run=False)
#     optimized_model(**val_examples[99].with_inputs())
#     lm.inspect_history(n=1)
#     evaluate_model(val_examples, lm, optimized_model)

def mipro_intent_classifier(base_model: dspy.Module, train_samples, mipro_mode):
    print(f"Base model: {base_model}")
    print(f"Total train size: {len(train_samples)}")
    print(f"Mipro mode: {mipro_mode}")
    mipro = dspy.MIPROv2(metric=validate_answer, auto=mipro_mode)
    optimized_model = mipro.compile(base_model, trainset=train_samples, max_bootstrapped_demos=3, max_labeled_demos=4,
                                    requires_permission_to_run=False)
    optimized_model(**val_examples[99].with_inputs())
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, optimized_model)


if __name__ == '__main__':
    # copro_intent_classifier()
    # copro_cot_intent_classifier()
    # mipro_intent_classifier(dspy.Predict(IntentClassifier), small_train_set, "light")
    mipro_intent_classifier(dspy.Predict(IntentClassifier), medium_train_set, "light")
    mipro_intent_classifier(dspy.Predict(IntentClassifier), create_examples_from_set('train', n=400), "light")
    # mipro_intent_classifier(dspy.ChainOfThought(IntentClassifier))

