import random
from typing import List

import dspy
from dspy import LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch, KNNFewShot

from chapter_3.dspy_structures import ClosedIntentSignature
from chapter_4.evaluate_dspy import validate_answer
from chapter_5.utils import evaluate_model, create_examples_from_set
from sentence_transformers import SentenceTransformer

gpt_4o_mini = dspy.LM("gpt-4o-mini", cache=False)
dspy.settings.configure(lm=gpt_4o_mini)
val_examples = create_examples_from_set('test', 876)
train_examples = create_examples_from_set('train', 100)


def labelled_few_shot(num_examples=10, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    intent_classifier = dspy.Predict(ClosedIntentSignature)

    labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
    few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier, trainset=train_examples)
    evaluate_model(val_examples, lm, few_shot_model)


def labelled_few_shot_cot(num_examples=10, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)
    labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
    few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    evaluate_model(val_examples, lm, few_shot_model)


def bootstrap_few_shot_cot(max_bootstrapped=4, max_labeled=10, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)

    optimizer = BootstrapFewShot(
        validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=1
    )

    bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    evaluate_model(val_examples, lm, bootrstrap_few_shot)


def bootstrap_few_shot(max_bootstrapped=4, max_labeled=10, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    intent_classifier_cot = dspy.ChainOfThought(ClosedIntentSignature)

    optimizer = BootstrapFewShot(
        validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=1
    )

    bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, bootrstrap_few_shot)


def bootstrap_few_shot_with_random_search(max_labeled, max_bootstrapped, max_candidate_programs, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
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
    evaluate_model(val_examples, lm, intent_classifier_bootstrap_few_shot_with_random_search)

def knn_example(module: dspy.Module,
                knn_train_examples: List[dspy.Example],
                embedding_model="sentence-transformers/all-MiniLM-L12-V2",
                max_labeled_demos=4,
                k=3,
                lm=gpt_4o_mini):
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
    evaluate_model(val_examples, lm, optimized_module)

if __name__ == '__main__':
    for lm in [dspy.LM("openai/gpt-4o-mini", cache=False), dspy.LM("openai/gpt-4.1-nano", cache=False)]:
        for num_examples in [10, 25, 50]:
            labelled_few_shot(num_examples=num_examples, lm=lm)
            labelled_few_shot_cot(num_examples=num_examples, lm=lm)

        for max_labeled, max_bootstrapped in [(10, 5), (25, 15), (0, 25)]:
            bootstrap_few_shot_cot(max_labeled=max_labeled, max_bootstrapped=max_bootstrapped, lm=lm)

        for max_labeled, max_bootstrapped, max_candidate_programs in [(25, 15, 5), (25, 15, 10), (25, 15, 15)]:
            bootstrap_few_shot_with_random_search(max_labeled=max_labeled, max_bootstrapped=max_bootstrapped,
                                                  max_candidate_programs=max_candidate_programs,
                                                  lm=lm)

        for examples in [create_examples_from_set('train', 100), create_examples_from_set('train', 500)]:
            for k in [5, 10, 15]:
                for embedding_model in ['sentence-transformers/all-MiniLM-L12-V2', 'BAAI/bge-base-en-v1.5']:
                    knn_example(dspy.ChainOfThought(ClosedIntentSignature), examples, embedding_model=embedding_model,
                                max_labeled_demos=k, k=k, lm=lm)
