import time
from tqdm import tqdm
import numpy as np

import dspy
from openai import OpenAI

from chapter_4.evaluate_dspy import validate_answer


def classify_message_prompt(message: str, system_prompt: str):
    model = 'gpt-4o-mini'
    client = OpenAI()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

