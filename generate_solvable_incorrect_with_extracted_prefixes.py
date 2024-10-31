import pandas as pd
from utils import get_naive_prefix



def extract_and_add_prefixes(df_solvable_incorrect: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe with solvable problems and their incorrect solutions, extract the prefixes from the incorrect solutions.
    """
    # Add a new "prefix" column to the dataframe
    df_solvable_incorrect['prefix'] = df_solvable_incorrect['candidate_solution'].apply(get_naive_prefix)

    return df_solvable_incorrect


def main():    
    input_filename = "datasets/cn_k12_math_problems_ss_command-r-plus-08-2024_10_5.csv"
    
    # Read the dataframe produced by the generate_straight_shot.py script.
    print(f"Loading dataframe from {input_filename}")
    df = pd.read_csv(input_filename)
    len_df = len(df)
    print(f"Loaded dataframe with {len_df} rows")

    # Now let's determine which problems are "solvable", and filter to those problems. Remember the list of solveable problem row_ids.
    print("Determining which problems are solvable...")
    completion_success_rates = df.groupby("row_id")["verification"].mean()
    solvable_problems_row_ids = completion_success_rates[(completion_success_rates >= 0.2) & (completion_success_rates <= 0.6)].index.tolist()
    n_solvable_problems = len(solvable_problems_row_ids)
    print(f"Found {n_solvable_problems} solvable problems out of {df["row_id"].nunique()} problems")

    print("Extracting incorrect solutions for solvable problems...")
    # Now let's filter to the incorrect solutions for these row_ids.
    df_solvable = df[df["row_id"].isin(solvable_problems_row_ids)]
    df_solvable_incorrect = df_solvable[~df_solvable["verification"]]
    print(f"Extracted {len(df_solvable_incorrect)} incorrect solutions for solvable problems")

    print("Extracting prefixes fromm incorrect solutions...")
    df_solvable_incorrect_with_prefixes = extract_and_add_prefixes(df_solvable_incorrect)
    print(f"Extracted prefixes from solveable problem incorrect solutions.")


    # Save both the list of solveable_problem row_ids (for off-policy prefix generation) and the incorrect solutions (for completion of remaining on-policy prefixes).
    # For the text filename, replace the ss (straight shot) with si (solvable incorrect) and include the number of solvable problems.
    output_txt_filename = input_filename.rsplit('_', 2)[0].replace("ss", "si") + f"_{n_solvable_problems}.txt"
    # For the df filename, replace the ss (straight shot) with sip (solvable incorrect prefixes) and include the number of solvable problems and number of incorrect prefixes
    output_csv_filename = input_filename.rsplit('_', 2)[0].replace("ss", "sip") + f"_{n_solvable_problems}_{len(df_solvable_incorrect_with_prefixes)}.csv"
    print(f"Saving output files {output_csv_filename} and {output_txt_filename}")
    df_solvable_incorrect_with_prefixes.to_csv(output_csv_filename, index=False)
    print(f"Saved incorrect solutions and prefixes for incorrect, solvable problems to {output_csv_filename}")
    with open(output_txt_filename, "w") as f:
        for row_id in solvable_problems_row_ids:
            f.write(f"{row_id}\n")
    print(f"Saved list of solvable problem row_ids to {output_txt_filename}")

    print("Done!")


if __name__ == "__main__":
    main()
