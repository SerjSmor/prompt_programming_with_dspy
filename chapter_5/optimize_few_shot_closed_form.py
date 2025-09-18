import random
from typing import List

import dspy
import pandas as pd
from datasets import load_dataset
from dspy import BootstrapFewShot, BootstrapFewShotWithRandomSearch, KNNFewShot, LM
from dspy.teleprompt import LabeledFewShot
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import mlflow

import common.utils
from utils import evaluate_model
from common.consts import GPT_4O_MINI, ATIS_INTENT_MAPPING
from chapter_3.dspy_structures import ClosedIntentSignature, IntentSignature
from chapter_4.evaluate_dspy import validate_answer

lm = LM('openai/gpt-4.1-nano', cache=False)

load_dotenv()

gpt_4o_mini = dspy.LM(GPT_4O_MINI, cache=False)
dspy.settings.configure(lm=gpt_4o_mini)


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

val_examples = create_examples_from_set('test', 876)
tiny_val_examples = create_examples_from_set('test', 10)
train_examples = create_examples_from_set('train', 100)


EXAMPLES_NO_LABELS = [dspy.Example(message='How do I get to st.Loise?').with_inputs('message'),
                      dspy.Example(message='What are the cheapest seats?').with_inputs('message'),
                      dspy.Example(message=train_examples[0]['message']).with_inputs('message')]


small_train_set = create_examples_from_set('train', 50)
medium_train_set = create_examples_from_set('train', 200)
tiny_train_set = create_examples_from_set('train', 10)

# the point of this example is to show the difference between a regular intent classifier prompt
# and a prompt that was optimized by labelled few shot, including examples
def labelled_few_shot(num_examples=10, lm=gpt_4o_mini, is_mlflow=False):
    dspy.settings.configure(lm=lm)
    print(f"lablled few shot: num examples: {num_examples}, lm={lm}")
    intent_classifier = dspy.Predict(ClosedIntentSignature)

    labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
    few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier, trainset=train_examples)
    # pred = intent_classifier(**train_examples[99].inputs())
    # print(pred)
    # lm.inspect_history(n=1)
    prediction = few_shot_model(**train_examples[99].inputs())
    print(prediction)
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, few_shot_model, is_mlflow, optimizer='labeled_few_shot', mlflow_logs={'num_examples': num_examples})
    print("***********************")


def labeled_few_shot_signature(lm: dspy.LM, signature: dspy.Signature, train_examples: list[dspy.Example], num_examples: int = 4):
    dspy.settings.configure(lm=lm)
    print(f"lablled few shot: num examples: {num_examples}, lm={lm}")
    intent_classifier = dspy.Predict(signature)

    labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
    few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier, trainset=train_examples)
    prediction = few_shot_model(**train_examples[99].inputs())
    print(prediction)
    lm.inspect_history(n=1)

def labelled_few_shot_cot(num_examples=10, lm=gpt_4o_mini, is_mlflow=False):
    dspy.settings.configure(lm=lm)
    # intent_classifier = dspy.Predict(IntentClassifier)
    print(f"labelled few shot cot: {num_examples}, lm={lm}")
    intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)
    labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
    few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    # prediction = few_shot_model(**train_examples[99].inputs())
    # lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, few_shot_model, is_mlflow, optimizer="labeled_few_shot",
                   mlflow_logs={'num_examples': num_examples})

def bootstrap_few_shot_cot(max_bootstrapped=4, max_labeled=10, lm=gpt_4o_mini, is_mlflow=False):
    print(f"bootstrap few shot cot, lm={lm}")
    dspy.settings.configure(lm=lm)
    intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)

    optimizer = BootstrapFewShot(
        validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=1
    )

    bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    # bootrstrap_few_shot(**train_examples[99].inputs())
    # lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, bootrstrap_few_shot, is_mlflow, optimizer="bootstrap_few_shot",
                   mlflow_logs={'max_labeled': max_labeled, 'max_bootstrapped': max_bootstrapped})
    

def bootstrap_few_shot(max_bootstrapped=4, max_labeled=10, lm=gpt_4o_mini, is_mlflow=False):
    dspy.settings.configure(lm=lm)
    print(f"Bootstrap few shot cot: bootstrapped {max_bootstrapped}, labeled {max_labeled}, lm={lm}")
    intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)

    optimizer = BootstrapFewShot(
        validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=1
    )

    bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    bootrstrap_few_shot(**train_examples[99].inputs())
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, bootrstrap_few_shot, is_mlflow, optimizer="bootstrap_few_shot",
                   mlflow_logs={'max_labeled': max_labeled, 'max_bootstrapped': max_bootstrapped})



def bootstrap_few_shot_with_random_search(max_labeled, max_bootstrapped, max_candidate_programs, lm=gpt_4o_mini, is_mlflow=False):
    dspy.settings.configure(lm=lm)
    print(f"Start bootstrap few with random search: {max_labeled}, {max_bootstrapped} {max_candidate_programs}, lm={lm}")
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        num_threads=10,
        num_candidate_programs=max_candidate_programs
    )

    intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)
    # create a val set from training set
    current_train_examples = create_examples_from_set('train', 4800)
    dev_set = random.Random().sample(current_train_examples, 100)

    for example in dev_set:
        current_train_examples.remove(example)

    intent_classifier_bootstrap_few_shot_with_random_search = optimizer.compile(intent_classifier_cot,
                                                                                trainset=current_train_examples,
                                                                                valset=dev_set)
    intent_classifier_bootstrap_few_shot_with_random_search(**train_examples[99].inputs())
    # lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, intent_classifier_bootstrap_few_shot_with_random_search, is_mlflow, optimizer="bootstrap_few_shot_with_random_search",
                   mlflow_logs={'max_labeled': max_labeled, 'max_bootstrapped': max_bootstrapped, 'candidate_programs': max_candidate_programs})
    
    print(f"End bootstrap few with random search: {max_labeled}, {max_bootstrapped} {max_candidate_programs}")


def knn_example(module: dspy.Module,
                knn_train_examples: List[dspy.Example],
                embedding_model="sentence-transformers/all-MiniLM-L12-V2",
                max_labeled_demos=4,
                k=3,
                lm=gpt_4o_mini, is_mlflow=False):

    print(f"knn. embedding model: {embedding_model}, k={k}, lm={lm.model}")
    dspy.settings.configure(lm=lm)
    # Initialize KNN with a sentence transformer model
    encoder_func = SentenceTransformer(embedding_model, token=False).encode

    knn_optimizer = KNNFewShot(
        k=k,
        trainset=knn_train_examples,
        vectorizer=dspy.Embedder(encoder_func),
        max_bootstrapped_demos=0,
        max_labeled_demos=max_labeled_demos
    )

    optimized_module = knn_optimizer.compile(module)
    evaluate_model(val_examples, lm, optimized_module, is_mlflow, optimizer="knn",
                   mlflow_logs={'max_labeled': max_labeled_demos, 'max_bootstrapped': 0, 'encoder': embedding_model,
                                'k': k, 'num_knn_train_examples': len(knn_train_examples)})


def compare_closed_open_signature_labeled_few_shot():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    open_label_examples = common.utils.create_examples_from_set('train', 4700)
    close_label_examples = create_examples_from_set('train', 4700)
    labeled_few_shot_signature(lm, IntentSignature, open_label_examples)
    labeled_few_shot_signature(lm, ClosedIntentSignature, close_label_examples)


def few_shot_optimization_evaluation(lm=gpt_4o_mini, is_mlflow=False):
    for num_examples in [10, 25, 50]:
        labelled_few_shot(num_examples=num_examples, lm=lm, is_mlflow=is_mlflow)
        labelled_few_shot_cot(num_examples=num_examples, lm=lm, is_mlflow=is_mlflow)

    for max_labeled, max_bootstrapped in [(10, 5), (25, 15), (0, 25)]:
        bootstrap_few_shot_cot(max_labeled=max_labeled, max_bootstrapped=max_bootstrapped, lm=lm, is_mlflow=is_mlflow)

    for max_labeled, max_bootstrapped, max_candidate_programs in [(25, 15, 5), (25, 15, 10), (25, 15, 15)]:
        bootstrap_few_shot_with_random_search(max_labeled=max_labeled, max_bootstrapped=max_bootstrapped,
                                              max_candidate_programs=max_candidate_programs,
                                              lm=lm, is_mlflow=is_mlflow)

    for examples in [create_examples_from_set('train', 100), create_examples_from_set('train', 500)]:
        for k in [5, 10, 15]:
            for embedding_model in ['sentence-transformers/all-MiniLM-L12-V2', 'BAAI/bge-base-en-v1.5']:
                print(f'Start KNN: {len(examples)}, k: {k}, {embedding_model}, max labeled: {k}')
                knn_example(dspy.ChainOfThought(ClosedIntentSignature), examples, embedding_model=embedding_model,
                            max_labeled_demos=k, k=k, lm=lm, is_mlflow=is_mlflow)
                print(f'End KNN: {len(examples)}, k: {k}, {embedding_model}, max labeled: {k}')


def baseline_experiment(intent_module: dspy.Module, lm: dspy.LM):
    evaluate_model(val_examples, lm, intent_module, is_mlflow=True)

def nano_baseline():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Create a unique name for your experiment.
    mlflow.set_experiment("Baseline gpt-4.1-nano")
    lm = dspy.LM("openai/gpt-4.1-nano", cache=False)
    baseline_experiment(dspy.Predict(ClosedIntentSignature), lm)
    baseline_experiment(dspy.ChainOfThought(ClosedIntentSignature), lm)

def gpt4omini_baseline():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Create a unique name for your experiment.
    mlflow.set_experiment("Baseline gpt-4o-mini")

    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    baseline_experiment(dspy.Predict(ClosedIntentSignature), lm)
    baseline_experiment(dspy.ChainOfThought(ClosedIntentSignature), lm)

def evaluate_models(is_mlflow: bool):
    if is_mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # Create a unique name for your experiment.
        mlflow.set_experiment("Prompt Example Optimization Test Set 4o-mini")
        # mlflow.autolog()

    few_shot_optimization_evaluation(lm, is_mlflow=True)


if __name__ == '__main__':
    nano_baseline()
    # compare_closed_open_signature_labeled_few_shot()

