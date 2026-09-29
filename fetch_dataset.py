import pandas as pd
from datasets import load_dataset

ds = load_dataset("Hello-SimpleAI/HC3", "all", trust_remote_code=True)["train"]

rows = []
for row in ds:
    for h in row["human_answers"]:
        if h and len(h.strip()) > 20:
            rows.append({"text": h.strip(), "label": "human"})
    for a in row["chatgpt_answers"]:
        if a and len(a.strip()) > 20:
            rows.append({"text": a.strip(), "label": "ai"})

df = pd.DataFrame(rows)
df = df.drop_duplicates(subset="text").sample(frac=1, random_state=42).reset_index(drop=True)

human_df = df[df.label == "human"]
ai_df = df[df.label == "ai"]
n = min(len(human_df), len(ai_df), 1000)

balanced = pd.concat([human_df.sample(n, random_state=42), ai_df.sample(n, random_state=42)])
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

balanced.to_csv("hc3_data.csv", index=False)
print(f"Saved {len(balanced)} rows -> hc3_data.csv")
print(balanced.label.value_counts())