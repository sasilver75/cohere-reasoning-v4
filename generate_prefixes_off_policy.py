import asyncio
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from utils import (
    generate_solution,
    generate_verification,
    get_naive_prefix,
    get_update_request_count,
    read_specific_rows,
)

completer_name = (  # Weak Completer
    "command-r-03-2024"  # Instruction-following conversational model (128k ctx)
)
strong_verifier_name = (
    "command-r-plus-08-2024"  # Most capable as of 10/12/2024 (128k ctx)
)

update_request_count = get_update_request_count(report_every_n_requests=10)


async def generate_incorrect_prefix(row: pd.Series, solution_idx: int) -> dict:
    """Generate a single incorrect solution and extract its prefix"""
    row_id = row["id"]  # Note: using 'id' instead of 'row_id' for original dataset
    problem = row["problem"]
    ground_truth_solution = row["solution"]  # Note: using 'solution' instead of 'ground_truth_solution'

    found_failure = False
    loop_count = 0
    
    while not found_failure:
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

        # Keep trying until we find an incorrect solution
        found_failure = not verification

    # Extract prefix from the incorrect solution
    prefix = get_naive_prefix(candidate_solution)

    return {
        "row_id": row_id,
        "problem": problem,
        "ground_truth_solution": ground_truth_solution,
        "solution_idx": solution_idx,
        "candidate_solution": candidate_solution,
        "verification_reasoning": verification_reasoning,
        "verification": verification,
        "prefix": prefix,
    }


async def process_batch(coroutines: list, batch_number: int, total_batches: int) -> list:
    """Process a batch of coroutines and return their results"""
    prefix_rows = []
    for coroutine in atqdm(
        asyncio.as_completed(coroutines),
        total=len(coroutines),
        desc=f"Processing batch {batch_number}/{total_batches}"
    ):
        result = await coroutine
        prefix_rows.append(result)
    return prefix_rows


async def main():
    BATCH_SIZE = 100
    n_prefixes_per_problem = 5  # Configurable number of prefixes to generate per problem
    
    # Read the row IDs from the text file
    print("Reading row IDs from text file...")
    with open("datasets/cn_k12_math_problems_sri_command-r-plus-08-2024_2.txt", "r") as f:
        row_ids = set(int(line.strip()) for line in f)
    print(f"Found {len(row_ids)} row IDs")

    # Read the corresponding rows from the original dataset
    print("Reading problems from original dataset...")
    original_df = read_specific_rows(
        source_filename="datasets/original/cn_k12_math_problems.csv",
        row_ids=row_ids
    )
    if original_df is None or len(original_df) == 0:
        print("Error: No matching rows found in original dataset")
        return
    print(f"Found {len(original_df)} matching problems")

    # Create all coroutines for prefix generation
    all_coroutines = []
    for _, row in original_df.iterrows():
        for solution_idx in range(n_prefixes_per_problem):
            all_coroutines.append(generate_incorrect_prefix(row, solution_idx))

    # Process coroutines in batches
    total_coroutines = len(all_coroutines)
    total_batches = (total_coroutines + BATCH_SIZE - 1) // BATCH_SIZE  # Round up division
    print(f"Processing {total_coroutines} total operations in {total_batches} batches of {BATCH_SIZE}")

    all_prefix_rows = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_coroutines)
        batch_coroutines = all_coroutines[start_idx:end_idx]
        
        # Process the batch
        prefix_rows = await process_batch(batch_coroutines, batch_idx + 1, total_batches)
        all_prefix_rows.extend(prefix_rows)
        
        # Optional: Add a small delay between batches to help with rate limiting
        if batch_idx < total_batches - 1:  # Don't delay after the last batch
            await asyncio.sleep(30)  # 1 second delay between batches

    # Combine all results
    final_df = pd.DataFrame(all_prefix_rows)
    final_df = final_df.sort_values(by=["row_id", "solution_idx"])

    # Save the results
    output_filename = f"datasets/cn_k12_math_problems_prefixes_off_policy_{completer_name}_{len(row_ids)}_{n_prefixes_per_problem}.csv"
    print(f"Saving {len(final_df)} prefixes to {output_filename}...")
    final_df.to_csv(output_filename, index=False)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
