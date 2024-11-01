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

update_request_count = get_update_request_count(report_every_n_requests=10)


async def verify_row_completion(row: pd.Series) -> VerificationResult:
    """
    Verify a single row of a dataframe.
    """
    problem = row["problem"]
    ground_truth_solution = row["ground_truth_solution"]
    row_id = row["row_id"]
    solution_idx = row["solution_idx"]
    completion_idx = row["completion_idx"]
    completion_solution = row["prefix"] + " " + row["completion"]

    return await generate_verification(
        row_id=row_id,
        problem=problem,
        ground_truth_solution=ground_truth_solution,
        candidate_solution=completion_solution,
        solution_idx=solution_idx,
        update_request_count=update_request_count,
        strong_verifier_name=verifier_name,
        completion_idx=completion_idx
    )


# TODO SAM: I think I have teh generate completions in parallel thing working, but I haven't wokred on the verification in parallel thing yet. That's a little harder because it requires fucking with VerificationResult and generate_verification in utils, which I don't know if I want to mess with.
# I think instead I'm just going to have a single completion script where I pass it the prefix column that I want to target, and it it will generate completions/verifications for just that column. And I'll run it four times.
async def verify_data(df: pd.DataFrame, prefix_columns: list[str]):
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
                "completion_idx": res.completion_idx,
                "completion_verification_reasoning": res.verification_reasoning,
                "completion_verification": res.verification,
            }
            for res in results
        ]
    )

    # Merge the verification results back to the original dataframe
    merged_df = pd.merge(
        df, 
        verification_df, 
        on=["row_id", "solution_idx", "completion_idx"],
        how="left", 
        sort=False
    )
    return merged_df


@retry(
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True,  # Reraises if all retries fail
)
def complete_row(row: pd.Series, prefix_col_name: str) -> str:
    """It seems that this is working well, errors are occurring about every 50 or so when processing 250, and recovering.
    The error isn't really printing out well though. I think it's because of timeouts.
    """
    problem = row["problem"]
    prefix = row[prefix_col_name]
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


def complete_row_multiple(row: pd.Series, n_completions: int, prefix_columns: list[str]) -> list[dict]:
    """
    Generate multiple completions for a single row.
    Returns a list of dictionaries, where each dictionary represents one round of completions
    (containing completions for all prefixes in that round).
    """
    # Each element is a completion for one of the prefixes.
    all_completion_rounds = []
    
    # For each completion round
    for completion_idx in range(n_completions):
        completion_round = {
            'completion_idx': completion_idx,
            'completions': {}  # Dict of new_completion_col_name: completion
        }
        
        # Generate completions for each prefix in this round
        for prefix_column_name in prefix_columns:  # e.g. "prefix_take_0.1"
            completion = complete_row(row, prefix_column_name)
            completion_round['completions'][f"{prefix_column_name}_completion"] = completion
        
        all_completion_rounds.append(completion_round)
    
    return all_completion_rounds


async def complete_data(df: pd.DataFrame, n_completions_per_prefix: int, prefix_columns: list[str]):
    """
    Generate n completions for each row in the dataframe.
    Returns a new dataframe where each row contains completions for all prefixes.
    """

    rows = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Completing {len(df)} row-solution prefixes with {n_completions_per_prefix} completions each (Sync)"
    ):
        completion_rounds = complete_row_multiple(row, n_completions_per_prefix, prefix_columns)
        
        # Create one new row for each completion round
        for completion_round in completion_rounds:
            new_row = row.copy()
            new_row["completion_idx"] = completion_round["completion_idx"]
            # Add all completions for this round to the row
            for col, completion in completion_round["completions"].items():
                new_row[col] = completion
            rows.append(new_row)

    return pd.DataFrame(rows)


async def main():
    is_on_policy = True
    n_completions_per_prefix = 5  # Configure number of completions per prefix

    source_filename = (
        "datasets/cn_k12_math_problems_prefixes_on_policy_command-r-plus-08-2024_191_3_take_0.3.csv"
    )
    output_suffix = "take_0.3"  # A bonus suffix to add to output filename, if needed for experiments

    # Load dataframe
    print(f"Loading dataframe from {source_filename}...")
    df = pd.read_csv(source_filename)
    # TESTING
    df = df.head(17)
    # TESTING
    len_df = len(df)
    print(f"Loaded dataframe with {len_df} row-solution prefixes in total!")

    # Find the number of solution_idxs per row_id
    n_prefixes_per_problem = df.groupby('row_id')['solution_idx'].nunique().iloc[0]
    print(f"Number of solution prefixes per row/problem: {n_prefixes_per_problem}")

    # Find the number of prefixes.
    prefix_columns = [col for col in df.columns if col.startswith("prefix_take_")]
    print(f"Detected {len(prefix_columns)} prefix columns: {prefix_columns}; All completions will be generated for these prefixes.")

    # Verify that this number is consistent across all row_ids
    if not (df.groupby('row_id')['solution_idx'].nunique() == n_prefixes_per_problem).all():
        raise ValueError("Inconsistent number of solution_idxs across row_ids -- This shouldn't happen")

    print(f"This should result in {len_df * n_completions_per_prefix} completions in total, times however many prefixes there are in t")

    # Process data in batches, accumulate and concatenate results (with checkpointing)
    # When setting bs, consider your rate limit of 500 requests/minute, including retries.
    # If you have 30 rows, and you're creating 5 completions and verifications per row, that's 30*5*2=300 requests/minute without retries. 
    # But that's just for a single prefix column. If you have 4 prefix columns, that's 30*5*2*4=1200 requests/minute without retries.
    # To get it under 500, we would need to set bs=10. 10*5*2*4=400, which is vaguely under the limit. For safety, let's even go smaller, down to 8.
    bs = 8
    print(f"Processing {len_df} rows in batches of {bs}")
    processed_dfs = []

    # We can naively batch this because each row-solution prefix can be independently completed and verified.
    for i in range(0, len_df, bs):
        batch_df = df.iloc[i : i + bs]
        print(f"Processing batch {i//bs + 1}. Records {i} to {min(i+bs, len_df)} of {len_df}")

        # Generate completions for the batch (sync)
        print(f"Generating completions for batch {i//bs + 1}...")
        completion_batch_df = await complete_data(batch_df, n_completions_per_prefix, prefix_columns)
        print(f"Finished generating completions for batch {i//bs + 1}")

        # Verify the completions for the batch (async)
        print(f"Verifying completions for batch {i//bs + 1}...")
        verified_batch_df = await verify_data(completion_batch_df, prefix_columns)
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
    output_filename = f"datasets/cn_k12_math_problems_completions_{completer_name}_{verified_df['row_id'].nunique()}_{n_prefixes_per_problem}_{n_completions_per_prefix}_{'ON' if is_on_policy else 'OFF'}{"_" + output_suffix if output_suffix else ""}.csv"

    print(f"Saving results to {output_filename}...")
    verified_df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}!")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())