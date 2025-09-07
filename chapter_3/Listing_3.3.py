df['intent'] = df['intent'].map(ATIS_INTENT_MAPPING) 
df = df.dropna(subset=['intent']) 
print(f"DF number of rows after removing multi label classes: {len(df)}")
