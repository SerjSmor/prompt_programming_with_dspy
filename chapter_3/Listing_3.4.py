unique_intents = df['intent'].unique() 
sorted_intents = sorted(unique_intents) 
numbered_list = "\n".join(f"{i + 1}. {intent}" 
                          for i, intent in enumerate(sorted_intents)) 
print(numbered_list)
