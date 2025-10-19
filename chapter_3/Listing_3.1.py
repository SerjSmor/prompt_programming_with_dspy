from datasets import load_dataset

ds = load_dataset("tuetschek/atis")
ds.set_format(type='pandas')
df = ds['test'][:]
