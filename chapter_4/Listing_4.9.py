def validate_answer(example: dspy.Example, prediction: dspy.Prediction, trace=None):
   return example.intent_label.lower() == prediction.intent_label.strip().lower()
