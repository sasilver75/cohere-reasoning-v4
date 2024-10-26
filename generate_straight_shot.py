import asyncio
import os
import pickle
import re
from collections import namedtuple

import pandas as pd
from cohere import AsyncClientV2
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

import prompts
from utils import (
    VerificationResult,
    extract_verification_from_response,
    get_update_request_count,
    read_specific_rows,
)

load_dotenv()
co = AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))

completer_name = (  # Weak Completer
    "command-r-03-2024"  # Instruction-following conversational model (128k ctx)
)
# completer_name = (  # Strong Completer
#     "command-r-plus-08-2024"  # Most capable as of 10/12/2024 (128k ctx)
# )
strong_verifier_name = (
    "command-r-plus-08-2024"  # Most capable as of 10/12/2024 (128k ctx)
)


update_request_count = get_update_request_count(report_every_n_requests=10)


async def generate_solutions(
    row_id: int, problem: str, n_solutions_per_problem: int
) -> list[str]:
    """
    Let's generate (n_solutions_per_problem) solutions for the problems. I'm assuming that this will result in at least one failure, given performance I've seen so far.
    """

    async def generate_solution(row_id: int, problem: str, solution_idx: str) -> str:
        retries_remaining = 5
        while retries_remaining:
            try:
                update_request_count("generate solution")
                response = await asyncio.wait_for(
                    co.chat(
                        model=completer_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompts.STRAIGHT_SHOT_SOLUTION_PROMPT.format(
                                    problem=problem
                                ),
                            }
                        ],
                        temperature=0.3,  # We want to use a "normal" t=.3 for this, because we want to evaluate how difficult problems are.
                    ),
                    timeout=60,
                )
                return response.message.content[0].text
            except asyncio.TimeoutError as e:
                retries_remaining -= 1
                print(
                    f"Timeout occurred when generating solution {solution_idx} for problem {row_id}. Retries remaining now {retries_remaining}."
                )
                if retries_remaining:
                    print("Retrying with {retries_remaining} retries remaining.")
                else:
                    # If this ever happens (which it shouldn't), let's raise the error so that everything falls over and I can complain to Eddie.
                    print(f"Fatal: Ran out of retries, reraising error.")
                    raise e
            except Exception as e:
                retries_remaining -= 1
                print(
                    f"Non-timeout exception occurred when generating solution {solution_idx} for problem {row_id}. Retries remaining now {retries_remaining}. Error: {e}"
                )
                if retries_remaining:
                    print(f"Retrying with {retries_remaining} retries remaining.")
                else:
                    print(f"Fatal: Ran out of retries, teraising error.")
                    raise e

    # Generaet n_solutions_per_problem solutions for this problem.
    solution_tasks = [
        generate_solution(row_id, problem, solution_idx)
        for solution_idx in range(n_solutions_per_problem)
    ]
    return await asyncio.gather(*solution_tasks)


async def generate_strong_verifications(
    row_id: int,
    problem: str,
    ground_truth_solution: str,
    candidate_solutions: list[str],
) -> list[VerificationResult]:
    """
    Given a question, ground-truth solution, andcollection of collection of candidate solutions
    Verify whether each candidate solution is correct, returning (trace, judgement, prefix) for each.

    CRITICALLY, this function gives back the verifications in the same order as the candidate solutions
    """

    async def generate_verification(
        row_id: int,
        problem: str,
        ground_truth_solution: str,
        candidate_solution: str,
        solution_idx: int,
    ) -> VerificationResult:
        retries_remaining = 5
        while retries_remaining:
            try:
                update_request_count("generate verification")
                response = await asyncio.wait_for(
                    co.chat(
                        model=strong_verifier_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompts.VERIFY_SOLUTION_PROMPT_WITH_PREFIX.format(
                                    problem=problem,
                                    solution=ground_truth_solution,
                                    candidate_solution=candidate_solution,
                                ),
                            }
                        ],
                        temperature=0.0,
                    ),
                    timeout=60,
                )
                return extract_verification_from_response(
                    response.message.content[0].text, row_id, solution_idx
                )
            except asyncio.TimeoutError as e:
                print(
                    f"Timeout occurred when generating verification {solution_idx} for problem {row_id}."
                )
                retries_remaining -= 1
                if retries_remaining:
                    print("Retrying with {retries_remaining} retries remaining.")
                else:
                    # If this ever happens (which it shouldn't), let's raise the error so that everything falls over and I can complain to Eddie.
                    print(f"Fatal: Ran out of retries, reraising error.")
                    raise e
            except (
                Exception
            ) as e:  # Note that Python only executes the first block that matches the raised exception, so there's no worry about TimeoutError double-decrementing retries.
                # I think the most likely reason for another exception is going to be some sort of parsing error in extraction, so let's just rerun that one.
                print(
                    f"A non-timeout exception occurred when generating verification {solution_idx} for problem {row_id}. Error: {type(e).__name__}: {e}"
                )
                retries_remaining -= 1
                if retries_remaining:
                    print(f"Retrying with {retries_remaining} retries remaining.")
                else:
                    print(f"Fatal: Ran out of retries, reraising error.")
                    raise e

    # Thing to note: I don't think it's necessarily going to be true that the solution_idx will be the same as the solution_idx in the generate_solutions call, since tasks might complete out of order.
    # TODO(SAM): Check this, or consider adding some sorting somehow, prior to returning results? I don't know.
    verification_tasks: list[VerificationResult] = [
        generate_verification(
            row_id, problem, ground_truth_solution, candidate_solution, solution_idx
        )
        for solution_idx, candidate_solution in enumerate(candidate_solutions)
    ]
    verification_results = await asyncio.gather(*verification_tasks)

    # Sort the results based on the solution_idx, so the list given back "matches" the list of candidate solutions.
    return sorted(verification_results, key=lambda x: x.solution_idx)


ProcessResult = namedtuple(
    "ProcessResult",
    [
        "row_id",
        "problem",
        "ground_truth_solution",
        "solutions",
        "solution_idxs",
        "verification_reasonings",
        "verifications",
        "prefix_reasonings",
        "prefixes",
    ],
)


async def process_row(row: pd.Series, n_solutions_per_problem: int) -> ProcessResult:
    problem = row["problem"]
    ground_truth_solution = row["solution"]
    row_id = row["id"]

    # Generate a number of solutions
    solutions: list[str] = await generate_solutions(
        row_id, problem, n_solutions_per_problem
    )

    # Generate a strong verification for each solution
    # CRITICALLY: THIS GIVES BACK THE VerificationResults IN THE SAME ORDER AS THE SOLUTIONS.
    strong_verifications: list[VerificationResult] = (
        await generate_strong_verifications(
            row_id, problem, ground_truth_solution, solutions
        )
    )
    verification_reasonings = [v.verification_reasoning for v in strong_verifications]
    verifications = [v.verification for v in strong_verifications]
    prefix_reasonings = [v.prefix_reasoning for v in strong_verifications]
    prefixes = [v.prefix for v in strong_verifications]
    solution_idxs = list(range(len(solutions)))

    return ProcessResult(
        row_id=row_id,
        problem=problem,
        ground_truth_solution=ground_truth_solution,
        solution_idxs=solution_idxs,
        solutions=solutions,
        verification_reasonings=verification_reasonings,
        verifications=verifications,
        prefix_reasonings=prefix_reasonings,
        prefixes=prefixes,
    )


async def process_data(df: pd.DataFrame, n_solutions_per_problem: int) -> pd.DataFrame:
    tasks = []
    for _, row in df.iterrows():
        tasks.append(process_row(row, n_solutions_per_problem))

    results: list[ProcessResult] = []
    for task in atqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Processing {len(df)} problems (async)",
    ):
        result = await task
        results.append(result)

    # Now that we have a list of process results, let's "unpack" it into a dataframe that we'd like to work on?
    rows = []
    for process_result in results:
        # Unpack each process result into a collection of rows, where each row describes a single solution and its verification.
        for (
            solution,
            verification,
            verification_reasoning,
            prefix_reasoning,
            prefix,
            solution_idx,
        ) in zip(
            process_result.solutions,
            process_result.verifications,
            process_result.verification_reasonings,
            process_result.prefix_reasonings,
            process_result.prefixes,
            process_result.solution_idxs,
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
                    "prefix_reasoning": prefix_reasoning,
                    "prefix": prefix,
                }
            )
    df = pd.DataFrame(rows)
    # Sort the DataFrame by row_id and solution_idx
    df = df.sort_values(by=["row_id", "solution_idx"])
    return df


async def main():
    # This is the number of problem/solution pairs to process; it will result in a dataframe with ~ n*10 rows.
    n_problems = 300  # n = None means all records
    n_solutions_per_problem = 5

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


async def main_solveable():
    """
    This is an alternative to main. The only difference should be that it selects SPECIFIC row_ids from the cnk12 dataframe, rather than some range.
    These row_ids are specified in a text file in datasets/, and represent those problems that seem to be solvable by the strong completer model who will complete these prefixes.
    """

    n_solutions_per_problem = 5

    source_filename = "datasets/original/cn_k12_math_problems.csv"

    # Get the solveable Ids
    ids = []
    with open("datasets/solveable_problems_row_ids.txt", "r") as f:
        for line in f:
            ids.append(int(line.strip()))

    # Get the specific rows
    df = read_specific_rows(source_filename, ids)
    len_df = len(df)
    print(f"Given {len(ids)} ids, found {len_df} rows")

    target_filename = f"datasets/cn_k12_math_problems_ss_solveable_problems_{completer_name}_{len_df}_{n_solutions_per_problem}.csv"

    # EVERYTHING BELOW HERE SHOULD BE THE SAME

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
        processed_dfs_filename = f"datasets/processed_ss_dfs_checkpoint.pkl"
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
    # asyncio.run(main())
    asyncio.run(main_solveable())
