from typing import List

import dspy
from dspy import BootstrapFewShot, Evaluate
from dspy.teleprompt import LabeledFewShot

from common.consts import GPT_4O_MINI
from common.utils import create_examples_from_set, validate_answer
from chapter_3.dspy_structures import IntentSignature

from dotenv import load_dotenv
load_dotenv()

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
intent_classifier = dspy.Predict(IntentSignature)
num_examples = 4

labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier, trainset=train_examples)
prediction = few_shot_model(**train_examples[100].inputs())
print(prediction)
lm.inspect_history(n=1)

