dspy.settings.configure(lm=lm, cache=False)
classifier = dspy.Predict(IntentSignature)
test_set_scores = []
for example in dev_set:
    examples = [example] * 10
    evaluator = dspy.Evaluate(devset=examples, num_threads=10, display_progress=False,
                              display_table=False, provide_traceback=False,
                              max_errors=100000)
    res = evaluator(classifier, metric=validate_answer)
    scores = [score for _, _, score in res.results]
    print(scores)
    test_set_scores.append(min(scores))
