import pandas as pd

import datasets

dataset_name = "AI-MO/NuminaMath-CoT"
dataset = datasets.load_dataset(dataset_name)

file_path = "datasets/original/cn_k12_math_problems.csv"

# Combine train and test sets (keep info), filter to cn_k12, add index
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])
train_df["set"] = "train"
test_df["set"] = "test"
df = pd.concat([train_df, test_df])

# Filter to only the cn_k12 subset
df = df[df["source"] == "cn_k12"]

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Set an explicit "id" column (since the dataset doesn't have one)
df["id"] = range(len(df))

# Save to csv
df.to_csv(file_path, index=False)
