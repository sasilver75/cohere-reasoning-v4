import re
import time
from collections import namedtuple
from typing import Optional

import pandas as pd
from pyparsing import deque

# VerificationResult includes the row_id and solution_idx to help with correctly ordering results
# to attach to the original dataframe.
VerificationResult = namedtuple(
    "VerificationResult",
    [
        "verification_reasoning",
        "verification",
        "prefix_reasoning",
        "prefix",
        "row_id",
        "solution_idx",
    ],
)


def extract_verification_from_response(
    verification_response: str,
    row_id: int,
    solution_idx: int,
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

    verification_prefix_reasoning_pattern = (
        r"<prefix_reasoning>(.*?)</prefix_reasoning>"
    )
    match = re.search(
        verification_prefix_reasoning_pattern, verification_response, re.DOTALL
    )
    if not match:
        print(
            f"Could not parse verification prefix reasoning for {verification_response}"
        )
        raise Exception(
            f"Could not parse verification prefix reasoning for {verification_response}"
        )
    verification_prefix_reasoning = match.group(1).strip()

    # Extract PREFIX
    verification_prefix_pattern = r"<verification_prefix>(.*?)</verification_prefix>"
    match = re.search(verification_prefix_pattern, verification_response, re.DOTALL)
    if not match:
        print(f"Could not parse verification prefix for {verification_response}")
        raise Exception(
            f"Could not parse verification prefix for {verification_response}"
        )
    verification_prefix = match.group(1).strip()

    return VerificationResult(
        verification_reasoning=verification_reasoning,
        verification=verified,
        prefix_reasoning=verification_prefix_reasoning,
        prefix=verification_prefix,
        row_id=row_id,
        solution_idx=solution_idx,
    )


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
