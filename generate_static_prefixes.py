import pandas as pd

from utils import read_specific_rows
from tqdm import tqdm
import random

def generate_static_prefix(row: pd.Series, idx: int) -> dict:
    """
    For a row from the original cn_k12 dataset, generate an "incorrect solution" and a static prefix
    randomly sampled from our list of static prefixes.

    Goal is to:
    row_id	problem	ground_truth_solution	solution_idx	candidate_solution	verification_reasoning	verification	prefix
    """
    # Read the static prefixes file and select a random prefix
    with open("static_prefixes.txt", "r") as f:
        prefixes = f.readlines()
    prefix = random.choice(prefixes).strip()

    # Assemble the dictionary representation of the row
    dict = {
        "row_id": row["id"],
        "problem": row["problem"],
        "ground_truth_solution": row["solution"],
        "solution_idx": idx,
        "candidate_solution": "[Solution Placeholder]", # This can just be a placeholder; doesn't need to be anything
        "verification_reasoning": "[Verification Reasoning Placeholder]", # This can just be a placeholder; doesn't need to be anything
        "verification": False, # We're not verifying these, so they're all "incorrect"
        "prefix": prefix
    }
    return dict


def main():
    """
    This can be sync.
    """
    # Path to the collection of solvable problem ids. We're going to create 
    input_filepath = "datasets/cn_k12_math_problems_si_command-r-plus-08-2024_191.txt"
    # TODO: NOTE: It might not make sense to have > 1 fixed prefixes per problem? Because the prefixes are all going to be different across solution_idx for a given row_id.
    # Though perhaps even with "random" prefixes, different problems might be more or less recoverable, so I guess it does make sense.
    n_prefixes_per_problem = 3

    # Read the row ids from the text file
    with open("datasets/cn_k12_math_problems_si_command-r-plus-08-2024_191.txt", "r") as f:
        row_ids = set(int(line.strip()) for line in f)
    print(f"Read {len(row_ids)} solvable problem row ids from {input_filepath}")

    # Read the related rows from cn_k12.
    original_df = read_specific_rows(
        source_filename="datasets/original/cn_k12_math_problems.csv",
        row_ids=row_ids
    )

    # For every row, we need to generate some {n_prefixes_per_problem} "incorrect solutions" and prefixes.
    acc = []
    for _, row in tqdm(original_df.iterrows(), total=len(original_df), desc="Generating static prefixes"):
        for idx in range(n_prefixes_per_problem):
            acc.append(generate_static_prefix(row, idx))
    
    # Create a dataframe from the accumulated results
    df = pd.DataFrame(acc)
    print(f"Generated {len(df)} static prefixes for {len(row_ids)} problems.")

    # Path to the output file.
    output_filepath = f"datasets/cn_k12_math_problems_prefixes_static_{len(row_ids)}_{n_prefixes_per_problem}.csv"
    df.to_csv(output_filepath, index=False)
    print(f"Saved static prefixes to {output_filepath}")


if __name__ == "__main__":
    main()