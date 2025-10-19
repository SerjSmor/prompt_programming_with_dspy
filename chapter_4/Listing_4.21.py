import time
import os

from dotenv import load_dotenv
import dspy
import mlflow

from chapter_3.dspy_structures import IntentSignature
from chapter_4.evaluate_dspy import validate_answer
from common.utils import create_examples_from_set

# os.environ['OPENAI_API_KEY']  = OPENAI_API_KEY

load_dotenv()

# Tell MLflow about the server URI.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("Open Intent Signature")
mlflow.autolog()


def evaluate_baseline(num_examples, model, module_type, num_threads, dataset="test"):
    lm = dspy.LM(model, cache=False)
    dspy.settings.configure(lm=lm)
    if module_type == 'cot':
        classifier = dspy.ChainOfThought(IntentSignature)
    else:
        classifier = dspy.Predict(IntentSignature)
    examples = create_examples_from_set(dataset, n=num_examples)
    evaluate_atis = dspy.Evaluate(devset=examples, num_threads=num_threads,
                                  display_progress=True, display_table=True,
                                  provide_traceback=True, max_errors=100000)
    start_time = time.time()
    classifier(message=examples[0].with_inputs())
    overall_score, _ = evaluate_atis(classifier, metric=validate_answer)
    end_time = time.time()
    print(f"total time: {end_time - start_time}")
    return overall_score


# for module_type in ['predict', 'cot']:

for module_type in ['cot']:
    # for model_name in ['openai/gpt-4o-mini', 'openai/gpt-3.5-turbo']:
    for model_name in ['openai/gpt-4o-mini']:
        # for _ in range(3):
        print()
        print(f"Model: {model_name}, Module: {module_type}")
        overall_score = evaluate_baseline(num_examples=100, model=model_name,
                                          module_type=module_type, num_threads=10)
        print(f"Overall score: {overall_score}")
