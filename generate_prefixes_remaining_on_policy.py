import asyncio
from types import coroutine
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from utils import generate_solution, generate_verification, get_naive_prefix, get_update_request_count

# completer_name = (  # Weak Completer
#     "command-r-03-2024"  # Instruction-following conversational model (128k ctx)
# )
completer_name = (  # Strong Completer
    "command-r-plus-08-2024"  # Most capable as of 10/12/2024 (128k ctx)
)
strong_verifier_name = (
    "command-r-plus-08-2024"  # Most capable as of 10/12/2024 (128k ctx)
)

update_request_count = get_update_request_count(report_every_n_requests=10)


async def persistently_get_incorrect_prefixes(row: pd.Series, n_prefixes_to_generate: int) -> pd.DataFrame:
    """
    Given a row, generate n additional prefixes. Return the new prefixes as their own dataframe.
    Later, we can then merge this with the sub_df of existing prefixes for the same problem, and reset solution_idxs.
    """

    async def _persistently_get_incorrect_prefix(row: pd.Series, idx: int):
        row_id = row["row_id"]
        problem = row["problem"]
        ground_truth_solution = row["ground_truth_solution"]
        solution_idx = 500 + idx  # This is a placeholder, solution_idxs will be reset later anyways.

        found_failure = False

        loop_count = 0
        while not found_failure:
            # This is just to make sure that we don't get stuck spending too much money. Likely requires manual examination afterwards as to why that was selected as a solvable problem.
            loop_count += 1
            if 10 < loop_count > 15:
                print(f"WARN: Failing to find an incorrect solution for row {row_id} after {loop_count} attempts")
            if loop_count > 15:
                print(f"FATAL: Failed to find an incorrect solution for row {row_id} after 15 attempts; Exiting.")
                raise Exception(f"Failed to find an incorrect solution for row {row_id} after 15 attempts")

            # Generate a candidate solution
            candidate_solution: str = await generate_solution(
                row_id=row_id,
                problem=problem,
                solution_idx=solution_idx,
                update_request_count=update_request_count,
                completer_name=completer_name
            )

            # Verify the candidate solution
            verification_result = await generate_verification(
                row_id=row_id,
                problem=problem,
                ground_truth_solution=ground_truth_solution,
                candidate_solution=candidate_solution,
                solution_idx=solution_idx,
                update_request_count=update_request_count,
                strong_verifier_name=strong_verifier_name
            )
            verification_reasoning = verification_result.verification_reasoning
            verification = verification_result.verification

            # Determine if we should keep trying for an incorrect solution
            found_failure = not verification

        # Now that we've found a failure, we extract the prefix.
        prefix = get_naive_prefix(candidate_solution)

        return {
            "row_id": row_id,
            "problem": problem,
            "ground_truth_solution": ground_truth_solution,
            "solution_idx": solution_idx,  # This is a placeholder, solution_idxs will be reset later anyways.
            "candidate_solution": candidate_solution,
            "verification_reasoning": verification_reasoning,
            "verification": verification,
            "prefix": prefix,
        }
    
    coroutines = [
        _persistently_get_incorrect_prefix(row, idx) 
        for idx in range(n_prefixes_to_generate)
    ]
    prefix_rows: list[dict] = []
    for coroutine in atqdm(
        asyncio.as_completed(coroutines),
        total=len(coroutines),
        desc=f"Processing {len(coroutines)} incorrect prefixes (async)"
    ):
        result = await coroutine
        prefix_rows.append(result)
    
    return pd.DataFrame(prefix_rows)


async def complete_prefixes(df: pd.DataFrame, n_incorrect_prefixes: int):
    """
    Given a dataframe of solvable problems with extracted prefixes, "pad out" the solved problems' prefixes to n_incorrect prefixes.
    """
    ...

    accumulated_padded_dfs = []
    for row_id in df["row_id"].unique():
        # Get the subset of the dataframe for this row_id
        sub_df = df[df["row_id"] == row_id]

        # Determine how many more prefixes we need to generate for this problem
        n_prefixes_to_generate = n_incorrect_prefixes - len(sub_df)

        # Get the first row from sub_df; This has some useful data for the new rows to construct
        first_row = sub_df.iloc[0]  # This should never error

        # Get a new dataframe of remaining prefixes for the problem
        new_prefix_df = await persistently_get_incorrect_prefixes(first_row, n_prefixes_to_generate)
        
        # Concatenate the sub_df and the new_prefix_df, then reset the solution_idxs..
        padded_sub_df = pd.concat([sub_df, new_prefix_df], ignore_index=True)
        padded_sub_df['solution_idx'] = range(len(padded_sub_df))
        
        # Add the combined dataframe to our accumulated results
        accumulated_padded_dfs.append(padded_sub_df)
    
    # Now that we have a df for every row_id, concatenate them all together.
    padded_df = pd.concat(accumulated_padded_dfs, ignore_index=True)

    # Let's just confirm that it's sorted by (row_id, solution_idx) by resorting
    padded_df = padded_df.sort_values(by=["row_id", "solution_idx"])

    return padded_df

        



# If we wanted to implement batching for this, it's going to be a little bit tricky. 
# We would just need to make sure that for every row_id in a batch, all existing solution_idx are present, from the dataframe.
async def main():
    # Note If n_incorrect_prefixes is already less than the number of incorrect prefixes that already exist for a given problem, we will just use the existing prefixes (# > n_incorrect_prefixes)
    n_incorrect_prefixes = 2

    # Input filename should point to a csv of a datframe with prefixes for incorrect, solvable problems.
    input_filename = ""
    output_filename = ""

    print(f"Reading dataframe from {input_filename}...")
    df = pd.read_csv(input_filename)
    print(f"Read dataframe with {len(df)} prefixes for {df["row_id"].nunique()} Numina problems")
    print(f"Completing prefixes for {len(df)} solvable problems...")

    print("Padding out prefixes...")
    df_with_padded_prefixes = await complete_prefixes(df, n_incorrect_prefixes)
    print(f"Completed padding out prefixes; Now {len(df_with_padded_prefixes)} prefixes for {df_with_padded_prefixes["row_id"].nunique()} Numina problems")

    print(f"Saving to {output_filename}...")
    df_with_padded_prefixes.to_csv(output_filename, index=False)
    print("Done!")





if __name__ == "__main__":
    asyncio.run(main())
