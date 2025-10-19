import time
from tqdm import tqdm

import numpy as np
from dotenv import load_dotenv
load_dotenv()

from sklearn.metrics import classification_report
import dspy
from dspy.evaluate import Evaluate

from chapter_3.dspy_structures import IntentSignature
from common.utils import create_examples_from_set

train_val_set = create_examples_from_set('train', n=100) #A
np.random.shuffle(train_val_set) #B
train_set = train_val_set[:20]
val_set = train_val_set[20:]

dev_test_set = create_examples_from_set('test', n=100) #C
np.random.shuffle(dev_test_set )
dev_set = dev_test_set[:50]
test_set = dev_test_set[50:]


def validate_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    return example.intent_label.lower() == prediction.intent_label.strip().lower()


def evaluate_baseline_without_dspy(dev_set: list[dspy.Example]):
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)  # A
    dspy.settings.configure(lm=lm)
    classifier = dspy.Predict(IntentSignature)  # B
    scores = []

    start_time = time.time()
    for example in tqdm(dev_set):  # C
        prediction = classifier(**example.inputs())
        score = validate_answer(example, prediction)
        scores.append(score)  # D

    end_time = time.time()
    average_score = sum(scores) / len(scores)  # E
    print(f"average score: {average_score}, total time: {end_time - start_time}")


def evaluate_baseline(dev_set: list[dspy.Example], model: str = "openai/gpt-4o-mini", num_threads=10):
    lm = dspy.LM(model, cache=False)
    dspy.settings.configure(lm=lm)
    classifier = dspy.Predict(IntentSignature) #B
    evaluator = dspy.Evaluate(devset=dev_set, #C
                              num_threads=num_threads,
                              display_progress=True,
                              display_table=5,
                              provide_traceback=True,
                              max_errors=100000)
    start_time = time.time()
    results = evaluator(classifier, metric=validate_answer) #D
    end_time = time.time()
    print(f"total time: {end_time - start_time}")
    return results.score


def evaluate_with_classification_report():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    dspy.settings.configure(lm=lm)
    classifier = dspy.Predict(IntentSignature)
    examples = create_examples_from_set('test') #A

    evaluator = Evaluate(devset=examples, num_threads=8, display_progress=True,
                         display_table=5, provide_traceback=True)
    start_time = time.time()
    res = evaluator(classifier, metric=validate_answer)

    predictions = []
    groundtruths = []
    for triplett in res.results: #B
        example, prediction, score = triplett
        groundtruths.append(example.intent_label)
        predictions.append(prediction.intent_label.strip("'")) #C

    end_time = time.time()
    print(f"total time: {end_time - start_time}")
    print(classification_report(groundtruths, predictions)) #D


if __name__ == '__main__':

    overall_score = evaluate_baseline(dev_set, num_threads=10)



