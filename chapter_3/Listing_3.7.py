import os
from openai import OpenAI

model = 'gpt-4o-mini'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI()
system_prompt = f'''
You are an expert of customer service intent classification. Your task is to classify the intent of customer messages of an airline company into one of the provided labels.                
Input: customer message
Output: One of the following classes: {",".join(unique_intents)}
'''
user_message = "I want to book a flight"
messages = [
{"role": "system", "content": system_prompt},
{"role": "user", "content": message}]
completion = client.responses.create(model=model, input=messages)
print(completion)
