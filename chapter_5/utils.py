import dspy
from dspy import Evaluate

from chapter_4.evaluate_dspy import validate_answer

def evaluate_model(examples: list[dspy.Example], lm: dspy.LM, classifier: dspy.Module):
    import time
    dspy.settings.configure(lm=lm)
    evaluate_atis = Evaluate(devset=examples, num_threads=10, display_progress=True, display_table=5,
                             max_errors=10000,
                             provide_traceback=True)
    start_time = time.time()
    evaluate_atis(classifier, metric=validate_answer)
    end_time = time.time()
    print(f"total time: {end_time - start_time}")