from sentence_transformers import SentenceTransformer
import dspy
from dspy import KNNFewShot

from chapter_3.dspy_structures import ClosedIntentSignature
from chapter_5.utils import create_examples_from_set

encoder_func = SentenceTransformer("sentence-transformers/all-MiniLM-L12-V2", token=False).encode


knn_train_examples = create_examples_from_set('train', 100)
train_examples = create_examples_from_set('train', 4700)
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)

knn_optimizer = KNNFewShot(
    k=3,
    trainset=knn_train_examples,
    vectorizer=dspy.Embedder(encoder_func),
    max_bootstrapped_demos=0,
    max_labeled_demos=4
)

optimized_module = knn_optimizer.compile(dspy.ChainOfThought(ClosedIntentSignature))
pred = optimized_module(**train_examples[199].with_inputs())
lm.inspect_history(n=1)

