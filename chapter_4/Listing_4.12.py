import time
import dspy
from tqdm import tqdm

lm = dspy.LM("openai/gpt-4o-mini", cache=False) 
dspy.settings.configure(lm=lm)
classifier = dspy.Predict(IntentSignature) 
scores = []
start_time = time.time()
for example in tqdm(dev_set): 
   prediction = classifier(**example.inputs())
   score = validate_answer(example, prediction)
   scores.append(score) 

end_time = time.time()
average_score = sum(scores) / len(scores) 
print(f"average score: {average_score}, total time: {end_time-start_time}")
