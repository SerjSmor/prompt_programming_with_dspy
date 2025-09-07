import dspy

def msgToIntentMetric(example, pred, trace=None):
    return pred.intent_label == example.intent_label

lm = dspy.LM("openai/gpt-4o-mini", api_key=OPENAI_API_KEY)
dspy.settings.configure(lm=lm)
labels = ['Cancel Subscription', 'Update Email', 'Refund Request', 
          'Bug Report', 'Unknown']
examples = [dspy.Example({'message': 'I want to end my subscription', 
                          'labels': labels,
                          'intent_label': 'CancelSubscription'})\
                    .with_inputs('message')] 
classifier = dspy.ChainOfThought('message, labels -> intent_label')
optimizer  = dspy.BootstrapFewShot(metric=msgToIntentMetric, max_labeled_demos=3)
classifier = optimizer.compile(classifier, trainset=examples)
prediction = classifier(
    message="My new email address is bob.smith@domain.com", labels=labels)
print(prediction)
