import asyncio
import functools
from types import coroutine
from numpy import take
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from utils import generate_solution, generate_verification, get_naive_prefix, get_update_request_count

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
            if loop_count > 10:
                print(f"WARN: Failing to find an incorrect solution for row {row_id} after {loop_count} attempts")
            if loop_count >= 35:
                print(f"FATAL: Failed to find an incorrect solution for row {row_id} after {loop_count} attempts; Exiting.")
                raise Exception(f"Failed to find an incorrect solution for row {row_id} after {loop_count} attempts")

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

        return {
            "row_id": row_id,
            "problem": problem,
            "ground_truth_solution": ground_truth_solution,
            "solution_idx": solution_idx,  # This is a placeholder, solution_idxs will be reset later anyways.
            "candidate_solution": candidate_solution,
            "verification_reasoning": verification_reasoning,
            "verification": verification,
        }
    
    coroutines = [
        _persistently_get_incorrect_prefix(row, idx) 
        for idx in range(n_prefixes_to_generate)
    ]
    prefix_rows: list[dict] = []
    for coroutine in atqdm(
        asyncio.as_completed(coroutines),
        total=len(coroutines),
        desc=f"Padding out {len(coroutines)} incorrect prefixes for row {row['row_id']} (async)"
    ):
        result = await coroutine
        prefix_rows.append(result)
    
    return pd.DataFrame(prefix_rows)


async def pad_incorrect_solutions(df: pd.DataFrame, n_incorrect_prefixes: int):
    """
    Given a dataframe of solvable problems with extracted prefixes, "pad out" the solved problems' prefixes to n_incorrect_prefixes.
    All generation happens concurrently across all row_ids.
    """
    # Create lists to store our generation tasks and their metadata
    generation_tasks = []
    
    # For each unique solvable row_id, create a coroutine to generate the remaining prefixes.
    for row_id in df["row_id"].unique():
        # Get the sub_df for a given row_id; these are the already-completed
        sub_df = df[df["row_id"] == row_id]
        n_prefixes_to_generate = n_incorrect_prefixes - len(sub_df)
        
        if n_prefixes_to_generate > 0:
            first_row = sub_df.iloc[0]
            generation_tasks.append({
                'coroutine': persistently_get_incorrect_prefixes(first_row, n_prefixes_to_generate),
                'row_id': row_id,
                'original_sub_df': sub_df
            })

    # Run all prefix generations concurrently
    new_prefix_dfs = []
    if generation_tasks:
        # Extract the coroutines
        coroutines = [task['coroutine'] for task in generation_tasks]
        for task, completed_coroutine in zip(generation_tasks, asyncio.as_completed(coroutines)):
            new_prefix_df = await completed_coroutine # TODO: HANDLING OF THIS NEEDS TO BE CHANGED?
            
            # Combine with original sub_df and reset solution_idxs
            padded_sub_df = pd.concat([task['original_sub_df'], new_prefix_df], ignore_index=True)
            padded_sub_df['solution_idx'] = range(len(padded_sub_df))  # "reset" the solution_idxs for the padded sub_df
            new_prefix_dfs.append(padded_sub_df)
    
    # Handle cases where no new prefixes were needed (they didn't make it into generation_tasks or new_prefix_dfs)
    for row_id in df["row_id"].unique():
        sub_df = df[df["row_id"] == row_id]
        if len(sub_df) >= n_incorrect_prefixes:
            # If we already have enough prefixes, just take the first n_incorrect_prefixes (This should never happen)
            sub_df = sub_df.head(n_incorrect_prefixes) # "reset" the solution_idxs for the padded sub_df
            sub_df['solution_idx'] = range(len(sub_df)) 
            new_prefix_dfs.append(sub_df)

    # Combine all results and make sure that we're sorted by (row_id, solution_idx)
    padded_df = pd.concat(new_prefix_dfs, ignore_index=True)
    padded_df = padded_df.sort_values(by=["row_id", "solution_idx"])
    
    return padded_df

async def generate_prefixes(df: pd.DataFrame, take_ps: list[float]) -> pd.DataFrame:
    """Given a dataframe of incorrect solutions, generate the prefixes for each solution"""
    # For each incorrect solution, generate len(take_ps) prefixes as new columns.
    for take_p in take_ps:
        print(f"Generating prefixes with take_p {take_p}...")
        get_take_p_prefix = functools.partial(get_naive_prefix, take=take_p)
        column_name = f"prefix_take_{take_p}"
        df[column_name] = df.apply(lambda row: get_take_p_prefix(row["candidate_solution"]), axis=1)
    return df

# If we wanted to implement batching for this, it's going to be a little bit tricky. 
# We would just need to make sure that for every row_id in a batch, all existing solution_idx are present, from the dataframe.
async def main():
    # Note If n_incorrect_prefixes is already less than the number of incorrect prefixes that already exist for a given problem, we will just use the existing prefixes (# > n_incorrect_prefixes)
    n_incorrect_solutions = 3
    take_ps = [0.1, 0.3, 0.5, 0.7]  # Determines the number and take of prefixes generated
    suffix_tag = f"take_{"_".join(map(str, take_ps))}"  # A bonus suffix to add to output filename, if needed for experiments

    # Input filename should point to a csv of a datframe with incorrect, solvable problems (no prefixes).
    input_filename = "datasets/cn_k12_math_problems_sip_command-r-plus-08-2024_191_636.csv"

    print(f"Reading dataframe from {input_filename}...")
    df = pd.read_csv(input_filename)
    print(f"Read dataframe with {len(df)} prefixes for {df["row_id"].nunique()} Numina problems")
    print(f"Completing prefixes for {len(df)} solvable problems...")

    # Pad out the incorrect solutions for each solvable problem
    print(f"Padding out incorrect solutions to {n_incorrect_solutions} per solveable problem...")
    df_with_padded_incorrect_solutions = await pad_incorrect_solutions(df, n_incorrect_solutions)
    print(f"Completed padding out incorrect solutions; Now {len(df_with_padded_incorrect_solutions)} solutions for {df_with_padded_incorrect_solutions["row_id"].nunique()} Numina problems")

    # Generate the n prefixes (based on take_ps) for each incorrect solution
    print(f"Adding prefixes to {len(df_with_padded_incorrect_solutions)} incorrect solutions using take_ps {take_ps}...")
    df_with_prefixes = await generate_prefixes(df_with_padded_incorrect_solutions, take_ps)
    print(f"Completed adding {len(take_ps)} prefixes for each incorrect solution; Now {len(df_with_prefixes)} prefixes in total for {df_with_prefixes["row_id"].nunique()} Numina problems")


    output_filename = f"datasets/cn_k12_math_problems_prefixes_on_policy_{completer_name}_{df_with_padded_incorrect_solutions['row_id'].nunique()}_{n_incorrect_solutions}{"_" + suffix_tag if suffix_tag else ""}.csv"

    print(f"Saving to {output_filename}...")
    df_with_prefixes.to_csv(output_filename, index=False)
    print("Done!")





if __name__ == "__main__":
    asyncio.run(main())
