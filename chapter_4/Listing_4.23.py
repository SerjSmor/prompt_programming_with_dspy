import time
import os

import pandas as pd
from dotenv import load_dotenv
import dspy
from datasets import load_dataset
from chapter_3.dspy_structures import IntentSignature, ClosedIntentSignature
from common.consts import ATIS_INTENT_MAPPING
import mlflow
# from chapter_4.evaluate_dspy import validate_answer
# from common.utils import create_examples_from_set

# os.environ['OPENAI_API_KEY']  = OPENAI_API_KEY

load_dotenv()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# Create a unique name for your experiment.
mlflow.set_experiment("Closed Intent Signature New Prompt")
mlflow.autolog()

def create_examples_from_set(set_name, n=-1):
    ds = load_dataset("tuetschek/atis")
    ds.set_format(type='pandas')
    df: pd.DataFrame = ds[set_name][:]
    df['intent'] = df['intent'].map(ATIS_INTENT_MAPPING)
    df = df.dropna(subset='intent')
    unique_intents = df['intent'].unique()
    if n > 0:
      df = df.sample(n=n, random_state=42) #A

    examples = []
    for index in df.index: #B
        row = df.loc[index]
        examples.append(
           dspy.Example(message=row['text'],
                        intent_label=row['intent']).with_inputs('message')
        )
    return examples


def validate_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    return example.intent_label.lower() == prediction.intent_label.strip().lower()

def evaluate_baseline(num_examples, model, module_type, num_threads, dataset="test"):
    lm = dspy.LM(model, cache=False)  
    dspy.settings.configure(lm=lm)
    if module_type == 'cot': 
      classifier = dspy.ChainOfThought(ClosedIntentSignature)
    else:
      classifier = dspy.Predict(ClosedIntentSignature)

    examples = create_examples_from_set(dataset, n=num_examples)
    classifier(message=examples[0].with_inputs())
    lm.inspect_history()
    evaluate_atis = dspy.Evaluate(devset=examples, num_threads=num_threads,
                                  display_progress=True, display_table=True,
                                  provide_traceback=True, max_errors=20)

    start_time = time.time()

    overall_score, results = evaluate_atis(classifier, metric=validate_answer)
    print(f"results: {results}")
    print(f"Failure score: {evaluate_atis.failure_score}, overall score: {overall_score}")
    end_time = time.time()
    print(f"total time: {end_time - start_time}")
    return overall_score

for module_type in ['cot', 'predict']:
    for model_name in ['openai/gpt-4o-mini']:
      # for _ in range(3):
        print()
        print(f"Model: {model_name}, Module: {module_type}")
        overall_score = evaluate_baseline(num_examples=876, model=model_name,
                                          module_type=module_type, num_threads=10)
        print(f"Overall score: {overall_score}")
