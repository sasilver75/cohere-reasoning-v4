import re
import time
from collections import namedtuple
from typing import Optional
import asyncio
from typing import Callable
import os

import pandas as pd
from pyparsing import deque
from dotenv import load_dotenv
from cohere import AsyncClientV2, Client
import prompts



load_dotenv()
co_async = AsyncClientV2(api_key=os.getenv("COHERE_API_KEY"))


def get_update_request_count(report_every_n_requests: int = 10):
    """
    A little utility function to help with rate limit debugging.
    Each time the outer function invokes you get a new closure, so you don't have to worry about resetting the count if you create two counters.

    Keeps track of the total number of invocations and the occurrences of invocations in the last minute.
    Reports stats every report_every_n_requests invocations.
    """
    count = 0
    timestamps = deque()

    def update(s: str):
        """
        "s" is a string to help with human debugging. You might pass "solution" or "verification" for instance.
        """
        nonlocal count
        count += 1
        current_time = time.time()
        timestamps.append(current_time)

        # Remove timestamps older than 60 seconds
        while timestamps and current_time - timestamps[0] > 60:
            timestamps.popleft()

        last_minute_count = len(timestamps)

        if count % report_every_n_requests == 0:
            print(
                f"{s} | Total requests: {count} | Requests in the last minute: {last_minute_count}"
            )

    return update


def read_specific_rows(
    source_filename: str, row_ids: set, chunksize: int = 10000
) -> Optional[pd.DataFrame]:
    """
    Reads specific rows from a CSV based on row_ids and writes them to a new CSV.

    Args:
        source_filename (str): Path to the source CSV file.
        target_filename (str): Path to save the filtered CSV.
        row_ids (set): Set of row_ids to filter.
        chunksize (int, optional): Number of rows per chunk. Defaults to 10000.
    """
    # Initialize a list to collect filtered chunks
    filtered_chunks = []

    # Iterate over the CSV file in chunks
    for chunk in pd.read_csv(source_filename, chunksize=chunksize):
        # Filter rows where 'row_id' is in the specified set (id in base csv == row_id in later csvs)
        filtered = chunk[chunk["id"].isin(row_ids)]
        if not filtered.empty:
            filtered_chunks.append(filtered)

    # Concatenate all filtered chunks
    return pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else None


def get_naive_prefix(solution: str) -> str:
    """
    Given a solution, return the first 30% of the solution, rounded to the nearest word.
    This is a naive way of getting a prefix; it's likely that prefix won't contain a perturbation.
    """
    words = solution.split()
    n_words = len(words)
    n_words_to_take = max(1, int(0.3 * n_words))
    return " ".join(words[:n_words_to_take])


async def generate_solution(row_id: int, problem: str, solution_idx: str, update_request_count: Callable[[str], None], completer_name: str) -> str:
        retries_remaining = 5
        while retries_remaining:
            try:
                update_request_count("generate solution")
                response = await asyncio.wait_for(
                    co_async.chat(
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


async def generate_solutions(
    row_id: int, problem: str, n_solutions_per_problem: int, update_request_count: Callable[[str], None], completer_name: str
) -> list[str]:
    """
    Let's generate (n_solutions_per_problem) solutions for the problems. I'm assuming that this will result in at least one failure, given performance I've seen so far.
    """

    # Generate n_solutions_per_problem solutions for this problem.
    coroutines = [
        generate_solution(row_id, problem, solution_idx, update_request_count, completer_name)
        for solution_idx in range(n_solutions_per_problem)
    ]
    return await asyncio.gather(*coroutines)




# VerificationResult includes the row_id and solution_idx to help with correctly ordering results
# to attach to the original dataframe.
VerificationResult = namedtuple(
    "VerificationResult",
    [
        "row_id",
        "solution_idx",
        "verification_reasoning",
        "verification",
        "completion_idx",
    ],
)

def extract_verification_from_response(
    verification_response: str,
    row_id: int,
    solution_idx: int,
    completion_idx: Optional[int] = 0  # This is only relevant for verifying completions, not solutions.
) -> VerificationResult:
    """
    Given a verification response, return whether the verifiation response indicates that the candidate solution was correct.
    There shouldn't be any extraction errors. If there's a problem, we should raise an exception (which, outside, will trigger a retry).
    """
    # Extract REASONING
    verification_reasoning_pattern = (
        r"<verification_reasoning>(.*?)</verification_reasoning>"
    )
    match = re.search(verification_reasoning_pattern, verification_response, re.DOTALL)
    if not match:
        print(f"Could not parse verification reasoning for {verification_response}")
        raise Exception(
            f"Could not parse verification reasoning for {verification_response}"
        )
    verification_reasoning = match.group(1).strip()

    # Extract RESULT
    verification_pattern = r"<verification_result>(.*?)</verification_result>"
    match = re.search(verification_pattern, verification_response, re.DOTALL)
    if not match:
        print(f"Could not parse verification result for {verification_response}")
        raise Exception(
            f"Could not parse verification result for {verification_response}"
        )
    verified = match.group(1).strip().lower() == "correct"

    return VerificationResult(
        row_id=row_id,
        solution_idx=solution_idx,
        verification_reasoning=verification_reasoning,
        verification=verified,
        completion_idx=completion_idx,
    )


async def generate_verification(
        row_id: int,
        problem: str,
        ground_truth_solution: str,
        candidate_solution: str,
        solution_idx: int,
        update_request_count: Callable[[str], None],
        strong_verifier_name: str,
        completion_idx: Optional[int] = 0  # This is only relevant for verifying completions, not solutions.
    ) -> VerificationResult:
        retries_remaining = 5
        while retries_remaining:
            try:
                update_request_count("generate verification")
                response = await asyncio.wait_for(
                    co_async.chat(
                        model=strong_verifier_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompts.VERIFY_SOLUTION_PROMPT.format(
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
                    response.message.content[0].text, row_id, solution_idx, completion_idx
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



async def generate_verifications(
    row_id: int,
    problem: str,
    ground_truth_solution: str,
    candidate_solutions: list[str],
    update_request_count: Callable[[str], None], 
    strong_verifier_name: str
) -> list[VerificationResult]:
    """
    Given a question, ground-truth solution, andcollection of collection of candidate solutions
    Verify whether each candidate solution is correct, returning (verification reasoning, verification) for each.

    CRITICALLY, this function gives back the verifications in the same order as the candidate solutions
    """

    # Verifications may be out of order, since coroutines might complete out of order.
    coroutines: list[VerificationResult] = [
        generate_verification(
            row_id, problem, ground_truth_solution, candidate_solution, solution_idx, update_request_count, strong_verifier_name
        )
        for solution_idx, candidate_solution in enumerate(candidate_solutions)
    ]
    verification_results = await asyncio.gather(*coroutines)

    # Sort the results based on the solution_idx, so the list given back "matches" the list of candidate solutions.
    return sorted(verification_results, key=lambda x: x.solution_idx)