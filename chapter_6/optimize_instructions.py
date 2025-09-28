import random

import dspy
from dotenv import load_dotenv
import mlflow
from common.consts import GPT_4O_MINI
from common.utils import create_examples_from_set
from chapter_4.evaluate_dspy import validate_answer
from utils import evaluate_model
from chapter_3.dspy_structures import IntentSignature as IntentClassifier, ClosedIntentSignature

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


def copro_cot_intent_classifier(lm: dspy.LM, breadth=4, depth=4, is_mlflow=False):
    dspy.settings.configure(lm=lm)
    cot_intent_classifier = dspy.ChainOfThought(ClosedIntentSignature)
    copro = dspy.COPRO(metric=validate_answer, breadth=breadth, depth=depth, track_stats=True)
    optimized_cot_intent_classifier = copro.compile(cot_intent_classifier, trainset=small_train_set,
                                                    eval_kwargs={"num_threads": 3})
    print(optimized_cot_intent_classifier.results_best)
    print(optimized_cot_intent_classifier.results_latest)
    print(optimized_cot_intent_classifier.total_calls)
    optimized_cot_intent_classifier(**val_examples[99].with_inputs())
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, optimized_cot_intent_classifier, is_mlflow=is_mlflow,
                   optimizer=dspy.COPRO.__name__,
                   mlflow_args={
                       "breadth": breadth,
                       "depth": depth
                   })


# def mipro_intent_classifier(base_model: dspy.Module):
#     print(f"Base model: {base_model}")
#     mipro = dspy.MIPROv2(metric=validate_answer, auto="light")
#     optimized_model = mipro.compile(base_model, trainset=small_train_set, max_bootstrapped_demos=3, max_labeled_demos=4,
#                                     requires_permission_to_run=False)
#     optimized_model(**val_examples[99].with_inputs())
#     lm.inspect_history(n=1)
#     evaluate_model(val_examples, lm, optimized_model)

def mipro_intent_classifier(lm: dspy.LM, base_model: dspy.Module, train_samples, val_samples, mipro_mode, max_labeled=3,
                            max_bootstraped=4,
                            is_mlflow=False):
    dspy.settings.configure(lm=lm)
    print(f"Base model: {base_model}")
    print(f"Total train size: {len(train_samples)}")
    print(f"Mipro mode: {mipro_mode}")
    mipro = dspy.MIPROv2(metric=validate_answer, auto=mipro_mode)
    optimized_model = mipro.compile(base_model, trainset=train_samples,
                                    valset=val_samples,
                                    max_bootstrapped_demos=max_bootstraped, max_labeled_demos=max_labeled,
                                    requires_permission_to_run=False)
    optimized_model(**val_examples[99].with_inputs())
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, optimized_model, is_mlflow=is_mlflow, optimizer=dspy.MIPROv2.__name__,
                   mlflow_args={
                       "mode": mipro_mode,
                       "max_bootstrapped": max_bootstraped,
                       "max_labeled": max_labeled
                   })


def optimize():
    train_samples = create_examples_from_set('train', 4800)
    dev_set = random.Random().sample(train_samples, 100)

    for lm in [dspy.LM("openai/gpt-4o-mini", cache=False), dspy.LM("openai/gpt-4.1-nano", cache=False)]:
        cot_intent_classifier = dspy.ChainOfThought(ClosedIntentSignature)
        # for breadth, depth in [(2, 2), (4, 4), (6, 6)]:
        for breadth, depth in [(6, 6)]:
            copro_cot_intent_classifier(lm, breadth, depth, is_mlflow=True)

        for mode in ["light", "medium", "heavy"]:
            for max_labeled, max_bootstrapped in [(10, 5), (25, 15), (0, 25)]:
                mipro_intent_classifier(lm, cot_intent_classifier, train_samples=train_samples, val_samples=dev_set,
                                        mipro_mode=mode, max_labeled=max_labeled, max_bootstraped=max_bootstrapped,
                                        is_mlflow=True)

if __name__ == '__main__':
    is_mlflow = True

    if is_mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Create a unique name for your experiment.
        mlflow.set_experiment("Instruction optimization")
        # mlflow.autolog()

    optimize()
