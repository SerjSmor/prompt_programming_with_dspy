from datasets import load_dataset
import pandas as pd

ds = load_dataset("tuetschek/atis")
ds.set_format(type='pandas')
df = ds['test'][:]
