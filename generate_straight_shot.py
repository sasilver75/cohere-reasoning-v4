import asyncio
import pickle
from collections import namedtuple

import pandas as pd
from tqdm.asyncio import tqdm as atqdm

from utils import (
    VerificationResult,
    generate_solutions,
    generate_verifications,
    get_update_request_count,
)


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


ProcessResult = namedtuple(
    "ProcessResult",
    [
        "row_id",
        "problem",
        "ground_truth_solution",
        "solution_idxs",
        "solutions",
        "verification_reasonings",
        "verifications",
    ],
)

async def process_row(row: pd.Series, n_solutions_per_problem: int) -> ProcessResult:
    problem = row["problem"]
    ground_truth_solution = row["solution"]
    row_id = row["id"]

    # 1) Generate a number of solutions
    solutions: list[str] = await generate_solutions(
        row_id, problem, n_solutions_per_problem, update_request_count, completer_name
    )
    solution_idxs = list(range(len(solutions)))

    # 2) Generate a strong verification for each solution
    # CRITICALLY: THIS GIVES BACK THE VerificationResults IN THE SAME ORDER AS THE SOLUTIONS.
    strong_verifications: list[VerificationResult] = (
        await generate_verifications(
            row_id, problem, ground_truth_solution, solutions, update_request_count, strong_verifier_name
        )
    )
    verification_reasonings = [v.verification_reasoning for v in strong_verifications]
    verifications = [v.verification for v in strong_verifications]

    return ProcessResult(
        row_id=row_id,
        problem=problem,
        ground_truth_solution=ground_truth_solution,
        solution_idxs=solution_idxs,
        solutions=solutions,
        verification_reasonings=verification_reasonings,
        verifications=verifications,
    )


async def process_data(df: pd.DataFrame, n_solutions_per_problem: int) -> pd.DataFrame:
    coroutines = []
    for _, row in df.iterrows():
        coroutines.append(process_row(row, n_solutions_per_problem))

    results: list[ProcessResult] = []
    for coroutine in atqdm(
        asyncio.as_completed(coroutines),
        total=len(coroutines),
        desc=f"Processing {len(df)} problems (async)",
    ):
        result = await coroutine
        results.append(result)

    # Now that we have a list of process results, let's "unpack" it into a dataframe that we'd like to work on?
    rows = []
    for process_result in results:
        # Unpack each process result into a collection of rows, where each row describes a single solution and its verification.
        for (  # There are the same N number of each of these.
            solution_idx,
            solution,
            verification_reasoning,
            verification,
        ) in zip(
            process_result.solution_idxs,
            process_result.solutions,
            process_result.verification_reasonings,
            process_result.verifications,
        ):
            rows.append(
                {
                    "row_id": process_result.row_id,
                    "problem": process_result.problem,
                    "ground_truth_solution": process_result.ground_truth_solution,
                    "solution_idx": solution_idx,
                    "candidate_solution": solution,
                    "verification_reasoning": verification_reasoning,
                    "verification": verification,
                }
            )
    df = pd.DataFrame(rows)

    # Sort the DataFrame by row_id and solution_idx
    df = df.sort_values(by=["row_id", "solution_idx"])
    
    return df


async def main():
    # This is the number of problem/solution pairs to process; it will result in a dataframe with ~ n*10 rows.
    n_problems = 5  # n = None means all records
    n_solutions_per_problem = 2

    source_filename = "datasets/original/cn_k12_math_problems.csv"
    target_filename = f"datasets/cn_k12_math_problems_ss_{completer_name}_{n_problems}_{n_solutions_per_problem}.csv"

    # Load the dataframe
    print(f"Loading dataframe from {source_filename}")
    df = (
        pd.read_csv(source_filename, nrows=n_problems)
        if n_problems is not None
        else pd.read_csv(source_filename)
    )
    len_df = len(df)
    print(f"Loaded dataframe with {len_df} rows")

    # Thinking: What should batch size be? We don't want to do 500/min rate limits... In here we're generating 2 calls per item (solution,verification), assume 3 with retries. 500/3 = 166 / 5 solutions per prolem = 30. Empirically 30 requests/minute peaks at about 250.
    bs = 30
    print(f"Processing {len_df} rows in batches of {bs}")
    processed_dfs = []
    for i in range(0, len_df, bs):
        batch_df = df.iloc[i : i + bs]
        print(f"Processing batch {i//bs + 1}. Records {i} to {i+bs} of {len_df}")
        batch_processed_df = await process_data(batch_df, n_solutions_per_problem)
        processed_dfs.append(batch_processed_df)

        # Save the processed_dfs list as a pickle file (overwriting any existing checkpoint)
        processed_dfs_filename = f"datasets/processed_dfs_checkpoint.pkl"
        with open(processed_dfs_filename, "wb") as f:
            pickle.dump(processed_dfs, f)
        print(
            f"Saved processed_dfs up to batch {i//bs + 1} to {processed_dfs_filename}"
        )

        # Wait for a minute before processing the next batch (assuming we're not on the last batch)
        if i + bs < len_df:
            print("Waiting for 60 seconds before processing the next batch...")
            await asyncio.sleep(60)

        print(f"Finished processing batch {i}")

    processed_df = pd.concat(processed_dfs, ignore_index=True)
    print(
        f"Finished processing {len_df} problems into {len(processed_df)} problem/solution pairs"
    )

    # Save results to CSV
    print(f"Saving results to {target_filename}")
    processed_df.to_csv(target_filename, index=False)
    print(f"Saved results to {target_filename}")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())