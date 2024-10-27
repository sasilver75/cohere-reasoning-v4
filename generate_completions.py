import asyncio
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from cohere import AsyncClientV2, Client
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

import prompts
from utils import (
    VerificationResult,
    generate_verification,
    get_update_request_count,
)

load_dotenv()
key = os.getenv("COHERE_API_KEY")
co_sync = Client(key)
co_async = AsyncClientV2(key)

completer_name = "command-r-plus-08-2024"  # Strong Completer - Most capable as of 10/12/2024 (128k ctx)
verifier_name = "command-r-plus-08-2024"  # Strong Verifier - Most capable as of 10/12/2024 (128k ctx)

update_request_count = get_update_request_count(report_every_n_requests=1)


async def verify_row_completion(row: pd.Series) -> VerificationResult:
    """
    Verify a single row of a dataframe.
    """
    problem = row["problem"]
    ground_truth_solution = row["ground_truth_solution"]
    row_id = row["row_id"]
    solution_idx = row["solution_idx"]
    # The full solution is the erroneous prefix + strong completer completion.
    completion_solution = row["prefix"] + " " + row["completion"]

    return await generate_verification(row_id, problem, ground_truth_solution, completion_solution, solution_idx, update_request_count, verifier_name)


async def verify_data(df: pd.DataFrame):
    """
    Generate verifications for candidate solutions in each row of a dataframe.
    We can do this asynchronously, using co_async instead of co_sync.
    """
    # Create coroutines (doesn't schedule them yet)
    coroutines = [verify_row_completion(row) for _, row in df.iterrows()]

    # Kick off and collect results
    results: list[VerificationResult] = []
    for task in atqdm(
        asyncio.as_completed(coroutines),
        total=len(coroutines),
        desc=f"Verifying {len(df)} completions (Async)",
    ):
        result = await task
        results.append(result)


    # Create a DataFrame from verification results
    verification_df = pd.DataFrame(
        [
            {
                "row_id": res.row_id,
                "solution_idx": res.solution_idx,
                "completion_verification_reasoning": res.verification_reasoning,
                "completion_verification": res.verification,
            }
            for res in results
        ]
    )

    # Merge the verification results back to the original dataframe (stick it to the side, matching in (row_id, solution_idx)
    merged_df = pd.merge(
        df, verification_df, on=["row_id", "solution_idx"], how="left", sort=False
    )
    return merged_df


@retry(
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,  # Reraises if all retries fail
)
def complete_row(row: pd.Series) -> str:
    """It seems that this is working well, errors are occurring about every 50 or so when processing 250, and recovering.
    The error isn't really printing out well though. I think it's because of timeouts.
    """
    problem = row["problem"]
    prefix = row["prefix"]
    row_id = row["row_id"]
    solution_idx = row["solution_idx"]

    user_turn = prompts.COMPLETION_PROMPT_USER.format(problem=problem)
    assistant_turn = prompts.COMPLETION_PROMPT_ASSISTANT.format(prefix=prefix)

    def get_completion():
        """Wrapper fn to pass to ThreadPoolExecutor.submit"""
        update_request_count("generate completion")
        return co_sync.chat(
            model=completer_name,
            message=prompts.COMPLETION_TEMPLATE.format(
                user_turn=user_turn,
                assistant_turn=assistant_turn,
            ),
            raw_prompting=True,
        )

    try:
        # Allows me to implement a timeout, which will then get handled by the tenacity @retry decorator.
        with ThreadPoolExecutor() as executor:
            future = executor.submit(get_completion)
            completion = future.result(timeout=60)
    except Exception as e:
        print(
            f"Unexpected error generating completion for row {row_id} (solution {solution_idx}): {e}"
        )
        if isinstance(e, TimeoutError):
            print(f"Error above is an instance of TimeoutError")
        raise e

    return completion.text


# TODO: THIS JUST GENERATES A SINGLE COMPLETION FOR EACH ROW PREFIX. DO WE WANT TO GENERATE MORE?
def complete_data(df: pd.DataFrame):
    """
    Generate completions for each row in the dataframe.
    We have to do this synchronously, since the v1 cohere API that supports raw prompting doesn't support async.
    """
    new_df = df.copy()

    completions = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Completing {len(df)} rows (Sync)"
    ):
        completions.append(complete_row(row))

    new_df["completion"] = completions

    return new_df


async def main():
    # n_problems = None  # Number of questions. questions = None means all records;
    # n_solutions_per_q = 5
    is_on_policy = True

    source_filename = (
        "datasets/cn_k12_math_problems_prefixes_on_policy_command-r-plus-08-2024_2_5.csv"
    )

    # Load dataframe
    print(f"Loading dataframe from {source_filename}...")
    df = pd.read_csv(source_filename)
    len_df = len(df)
    print(f"Loaded dataframe with {len_df} rows!")

    # Find the number of solution_idxs per row_id
    n_prefixes_per_problem = df.groupby('row_id')['solution_idx'].nunique().iloc[0]
    print(f"Number of solutions per question: {n_prefixes_per_problem}")

    # Verify that this number is consistent across all row_ids
    if not (df.groupby('row_id')['solution_idx'].nunique() == n_prefixes_per_problem).all():
        raise ValueError("Inconsistent number of solution_idxs across row_ids")

    # Process data in batches, accumulate and concatenate results (with checkpointing)
    bs = 125  # This seems to work fine
    print(f"Processing {len_df} rows in batches of {bs}")
    processed_dfs = []

    for i in range(0, len_df, bs):
        batch_df = df.iloc[i : i + bs]
        print(f"Processing batch {i//bs + 1}. Records {i} to {i+bs} of {len_df}")

        # Generate completions for the batch (sync)
        print(f"Generating completions for batch {i//bs + 1}...")
        completion_batch_df = complete_data(batch_df)
        print(f"Finished generating completions for batch {i//bs + 1}")

        # Verify the completions for the batch (async)
        print(f"Verifying completions for batch {i//bs + 1}...")
        verified_batch_df = await verify_data(completion_batch_df)
        print(f"Finished verifying completions for batch {i//bs + 1}")

        processed_dfs.append(verified_batch_df)

        # Save checkpoint
        checkpoint_filename = f"datasets/processed_completion_dfs_checkpoint.pkl"
        with open(checkpoint_filename, "wb") as f:
            pickle.dump(processed_dfs, f)
        print(f"Saved processed_dfs up to batch {i//bs + 1} to {checkpoint_filename}")

        # Wait before processing the next batch (if not the last batch)
        if i + bs < len_df:
            print("Waiting for 60 seconds before processing the next batch...")
            await asyncio.sleep(60)

    verified_df = pd.concat(processed_dfs, ignore_index=True)
    print(f"Finished processing {len_df} rows into {len(verified_df)} verified rows")

    # Save results to CSV
    # NumberOfProblems_NumberOfPrefixesPerProblem_NumberOfCompletionsPerPrefix_IsOnPolicy
    output_filename = f"datasets/cn_k12_math_problems_completions_{completer_name}_{verified_df["row_id"].nunique()}_{n_prefixes_per_problem}_1_{"ON" if is_on_policy else "OFF"}.csv"

    print(f"Saving results to {output_filename}...")
    verified_df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}!")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
