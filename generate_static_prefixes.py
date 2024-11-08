import pandas as pd

from utils import read_specific_rows


def main():
    """
    This can be sync.
    """
    # Path to the collection of solvable problem ids. We're going to create 
    input_filepath = "datasets/cn_k12_math_problems_si_command-r-plus-08-2024_191.txt"
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

    # For every row, we need to generate some "incorrect solutions" and prefixes.
    


    # Path to the output file. TODO: Also add the length of the dataframe before
    output_filepath = f"datasets/cn_k12_math_problems_static_prefixes_{n_prefixes_per_problem}.txt"



if __name__ == "__main__":
    main()