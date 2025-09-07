from dotenv import load_dotenv

load_dotenv()
openai_lm = dspy.LM('openai/gpt-4o')
