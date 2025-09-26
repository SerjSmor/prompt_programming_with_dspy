import random
from typing import List

import dspy
from dspy import BootstrapFewShot, BootstrapFewShotWithRandomSearch, KNNFewShot, LM
from dspy.teleprompt import LabeledFewShot
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from utils import evaluate_model
from common.consts import GPT_4O_MINI
from common.utils import create_examples_from_set
from chapter_3.dspy_structures import IntentSignature as StringBasedIntentClassifier
from chapter_4.evaluate_dspy import validate_answer

lm = LM('openai/gpt-4.1-nano', cache=False)

load_dotenv()

gpt_4o_mini = dspy.LM(GPT_4O_MINI, cache=False)
dspy.settings.configure(lm=gpt_4o_mini)

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
def labelled_few_shot(num_examples=10, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    print(f"lablled few shot: num examples: {num_examples}, lm={lm}")
    intent_classifier = dspy.Predict(StringBasedIntentClassifier)

    labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
    few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier, trainset=train_examples)
    # pred = intent_classifier(**train_examples[99].inputs())
    # print(pred)
    # lm.inspect_history(n=1)
    prediction = few_shot_model(**train_examples[99].inputs())
    print(prediction)
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, few_shot_model)
    print("***********************")




def labelled_few_shot_cot(num_examples=10, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    # intent_classifier = dspy.Predict(IntentClassifier)
    print(f"labelled few shot cot: {num_examples}, lm={lm}")
    intent_classifier_cot = dspy.ChainOfThought(StringBasedIntentClassifier)
    labeled_few_shot_optimizer = LabeledFewShot(k=num_examples)
    few_shot_model = labeled_few_shot_optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    # prediction = few_shot_model(**train_examples[99].inputs())
    # lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, few_shot_model)

def bootstrap_few_shot_cot(max_bootstrapped=4, max_labeled=10, lm=gpt_4o_mini):
    print(f"bootstrap few shot cot, lm={lm}")
    dspy.settings.configure(lm=lm)
    intent_classifier_cot = dspy.ChainOfThought(StringBasedIntentClassifier)

    optimizer = BootstrapFewShot(
        validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=1
    )

    bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    # bootrstrap_few_shot(**train_examples[99].inputs())
    # lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, bootrstrap_few_shot)
    

def bootstrap_few_shot(max_bootstrapped=4, max_labeled=10, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    print(f"Bootstrap few shot cot: bootstrapped {max_bootstrapped}, labeled {max_labeled}, lm={lm}")
    intent_classifier_cot = dspy.ChainOfThought(StringBasedIntentClassifier)

    optimizer = BootstrapFewShot(
        validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        max_rounds=1
    )

    bootrstrap_few_shot = optimizer.compile(student=intent_classifier_cot, trainset=train_examples)
    bootrstrap_few_shot(**train_examples[99].inputs())
    lm.inspect_history(n=1)
    evaluate_model(val_examples, lm, bootrstrap_few_shot)



def bootstrap_few_shot_with_random_search(max_labeled, max_bootstrapped, max_candidate_programs, lm=gpt_4o_mini):
    dspy.settings.configure(lm=lm)
    print(f"Start bootstrap few with random search: {max_labeled}, {max_bootstrapped} {max_candidate_programs}, lm={lm}")
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=validate_answer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        num_threads=10,
        num_candidate_programs=max_candidate_programs
    )

    intent_classifier_cot = dspy.ChainOfThought(StringBasedIntentClassifier)
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
    evaluate_model(val_examples, lm, intent_classifier_bootstrap_few_shot_with_random_search)
    print(f"End bootstrap few with random search: {max_labeled}, {max_bootstrapped} {max_candidate_programs}")


def knn_example(module: dspy.Module,
                knn_train_examples: List[dspy.Example],
                embedding_model="sentence-transformers/all-MiniLM-L12-V2",
                max_labeled_demos=4,
                k=3,
                lm=gpt_4o_mini):

    print(f"knn. embedding model: {embedding_model}, k={k}, lm={lm.model}")
    dspy.settings.configure(lm=lm)
    # Initialize KNN with a sentence transformer model
    encoder_func = SentenceTransformer(embedding_model, token="").encode

    knn_optimizer = KNNFewShot(
        k=k,
        trainset=knn_train_examples,
        vectorizer=dspy.Embedder(encoder_func),
        max_bootstrapped_demos=0,
        max_labeled_demos=max_labeled_demos
    )

    optimized_module = knn_optimizer.compile(module)
    evaluate_model(val_examples, lm, optimized_module)



def few_shot_optimization_evaluation(lm=gpt_4o_mini):
    # for num_examples in [10, 25, 50]:
    #     labelled_few_shot(num_examples=num_examples, lm=lm)
    #     labelled_few_shot_cot(num_examples=num_examples, lm=lm)
    #
    # for max_labeled, max_bootstrapped in [(10, 5), (25, 15), (0, 25)]:
    #     bootstrap_few_shot_cot(max_labeled=max_labeled, max_bootstrapped=max_bootstrapped, lm=lm)
    #
    # for max_labeled, max_bootstrapped, max_candidate_programs in [(25, 15, 5), (25, 15, 10), (25, 15, 15)]:
    #     bootstrap_few_shot_with_random_search(max_labeled=max_labeled, max_bootstrapped=max_bootstrapped,
    #                                           max_candidate_programs=max_candidate_programs,
    #                                           lm=lm)

    for examples in [create_examples_from_set('train', 100), create_examples_from_set('train', 500)]:
        # for k in [5, 10, 15]:
        for k in [5]:
            for embedding_model in ['sentence-transformers/all-MiniLM-L12-V2', 'BAAI/bge-base-en-v1.5']:
                print(f'Start KNN: {len(examples)}, k: {k}, {embedding_model}, max labeled: {k}')
                knn_example(dspy.ChainOfThought(StringBasedIntentClassifier), examples, embedding_model=embedding_model,
                            max_labeled_demos=k, k=k, lm=lm)
                print(f'End KNN: {len(examples)}, k: {k}, {embedding_model}, max labeled: {k}')


def _nano_experiment_baseline(intent_module: dspy.Module):
    # Average Metric: 735.00 / 876 (83.9%): 100%|██████████| 876/876 [02:12<00:00,  6.63it/s]
    # baseline
    from dspy import LM
    lm = LM('openai/gpt-4.1-nano', cache=False)
    print(f"baseline nano experiment")
    evaluate_model(val_examples, lm, intent_module)
    print("end baseline nano experiment")

def nano_baseline():

    _nano_experiment_baseline(dspy.Predict(StringBasedIntentClassifier))
    _nano_experiment_baseline(dspy.ChainOfThought(StringBasedIntentClassifier))


def evaluate_models():
    lm = LM('openai/gpt-4.1-nano', cache=False)
    few_shot_optimization_evaluation(lm)

if __name__ == '__main__':
    labelled_few_shot(lm=lm)



