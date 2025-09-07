import os
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

lm_mini  = dspy.LM(model='gpt-4o-mini', cache=False, temperature=0.0)
lm_turbo = dspy.LM(model='gpt-3.5-turbo', cache=False, temperature=0.0)
dspy.settings.configure(lm=lm_turbo)
prog = dspy.Predict('question->answer')
prediction = prog(question='Write a 5-line poem about toasters') 
print(prediction)

with dspy.context(lm=lm_mini):
  prediction = prog(question='Write a 5-line poem about toasters') 
  print(prediction)

prog.set_lm(lm_turbo)
prediction = prog(question='Write a 5-line poem about toasters') 
print(prediction)
