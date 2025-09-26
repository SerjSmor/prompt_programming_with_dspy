from typing import List, Literal

import dspy
import pandas as pd


class ClosedIntentSignature(dspy.Signature):
    """
        Classify the message into one of the possible labels.
        """
    message: str = dspy.InputField()
    intent_label: Literal['Flight Booking Request',
                          'Airfare and Fees Questions',
                          'Ground Transportation Inquiry',
                          'Inquiry about In-flight Meals',
                          'Airport Information and Queries',
                          'Airline Information Request',
                          'Time Inquiry',
                          'Airport Location Inquiry',
                          'Ground Transportation Cost Inquiry',
                          'Flight Quantity Inquiry',
                          'Abbreviation and Fare Code Meaning Inquiry',
                          'Airport Distance Inquiry',
                          'Aircraft Type Inquiry',
                          'Aircraft Seating Capacity Inquiry',
                          'Flight Number Inquiry'
    ] = dspy.OutputField()


from typing import List

import dspy
from dspy import BootstrapFewShot, Evaluate
from dspy.teleprompt import LabeledFewShot
from datasets import load_dataset

from common.consts import GPT_4O_MINI, ATIS_INTENT_MAPPING
from common.utils import create_examples_from_set, validate_answer
from chapter_3.dspy_structures import IntentSignature

from dotenv import load_dotenv
load_dotenv()

def create_examples_from_set(set_name, n):
    ds = load_dataset("tuetschek/atis")
    ds.set_format(type='pandas')
    df: pd.DataFrame = ds[set_name][:]
    df['intent'] = df['intent'].map(ATIS_INTENT_MAPPING)
    df = df.dropna(subset='intent')
    if n > 0:
        df = df.sample(n=n, random_state=42)  # A

    examples = []
    for index in df.index:  # B
        row = df.loc[index]
        examples.append(
            dspy.Example(message=row['text'],
                         intent_label=row['intent']).with_inputs('message')
        )
    return examples

def evaluate_model(examples: List[dspy.Example], lm: dspy.LM, classifier: dspy.Module):
    import time
    dspy.settings.configure(lm=lm)
    evaluate_atis = Evaluate(devset=examples, num_threads=10, display_progress=True, display_table=5,
                             provide_traceback=True, return_outputs=True)
    start_time = time.time()
    evaluate_atis(classifier, metric=validate_answer)
    end_time = time.time()
    print(f"total time: {end_time - start_time}")


lm = dspy.LM(GPT_4O_MINI, cache=False)
dspy.settings.configure(lm=lm)
train_examples = create_examples_from_set('train', 4700)
intent_classifier = dspy.Predict(ClosedIntentSignature)
num_examples = 4

labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier, trainset=train_examples)
prediction = few_shot_model(**train_examples[100].inputs())
print(prediction)
lm.inspect_history(n=1)

