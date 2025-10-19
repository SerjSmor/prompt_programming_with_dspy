import time

import numpy as np
from tqdm import tqdm
import dspy
from datasets import load_dataset

from chapter_3.openai_structures import classify_message_prompt
from chapter_4.evaluate_dspy import validate_answer
from common.consts import ATIS_INTENT_MAPPING
from common.utils import create_examples_from_set

dev_set = create_examples_from_set('test', 10)


def evaluate_openai_manual_prompt(system_prompt, dev_set, unique_intents):
    scores = []
    start_time = time.time()
    for example in tqdm(dev_set):
        prediction = classify_message_prompt(example.message, system_prompt)
        score = validate_answer(example, dspy.Prediction(intent_label=prediction))
        scores.append(score)
    end_time = time.time()
    print(f"Accuracy: {np.mean(scores)}, total time: {end_time - start_time}")


if __name__ == '__main__':
    ds = load_dataset("tuetschek/atis")
    ds.set_format(type='pandas')
    df = ds['test'][:]

    df['intent'] = df['intent'].map(ATIS_INTENT_MAPPING)  # A
    df = df.dropna(subset=['intent'])  # B
    print(f"DF number of rows after removing multi label classes: {len(df)}")

    unique_intents = df['intent'].unique()  # A
    sorted_intents = sorted(unique_intents)  # B
    numbered_list = "\n".join(f"{i + 1}. {intent}"
                              for i, intent in enumerate(sorted_intents))  # C
    print(numbered_list)

    system_prompt = f'''You are an expert of intent classification. Your task is to classify the intent of customer messages of an airline company into one of the provided labels. 
        Input: customer message
        Output: One of the following classes: {",".join(unique_intents)}'''

    evaluate_openai_manual_prompt(system_prompt, dev_set, unique_intents)
