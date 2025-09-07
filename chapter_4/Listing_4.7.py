import dspy
turbo = dspy.LM(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)
generator = dspy.Predict("user_message -> alternative_wording")

synethetic_examples = []
for original_example in test_set:   
  for _ in range(2):    
    prediction = generator(user_message=original_example.message)
    synethetic_examples.append(
        dspy.Example(message=prediction.alternative_wording, 
                     labels=unique_intents,
                     intent_label=original_example.intent_label                     
                     ).with_inputs('message', 'labels')
    ) 
