import os
from dotenv import load_dotenv

load_dotenv()
print(os.environ['OPENAI_API_KEY'])
openai_lm = dspy.LM('openai/gpt-4o', api_key=os.environ['OPENAI_API_KEY'])
