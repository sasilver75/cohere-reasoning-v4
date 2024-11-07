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
from concurrent.futures import Future

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


async def verify_row_completion(row: pd.Series, prefix_column: str) -> VerificationResult:
    """
    Verify a single row of a dataframe.
    """
    problem = row["problem"]
    ground_truth_solution = row["ground_truth_solution"]
    row_id = row["row_id"]
    solution_idx = row["solution_idx"]
    completion_idx = row["completion_idx"]
    completion_solution = row[prefix_column] + " " + row[f"{prefix_column}_completion"]

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


async def verify_data(df: pd.DataFrame, prefix_column: str):
    """
    Generate verifications for candidate solutions in each row of a dataframe.
    We can do this asynchronously, using co_async instead of co_sync.
    """
    # Create coroutines (doesn't schedule them yet)
    coroutines = [verify_row_completion(row, prefix_column) for _, row in df.iterrows()]

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
                f"{prefix_column}_completion_verification_reasoning": res.verification_reasoning,
                f"{prefix_column}_completion_verification": res.verification,
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


def retry_callback(retry_state):
    # This is a fallback value used if we fail all retries because of (eg) timeouts.
    print(f"AAAAAAAH")
    return "~FAILED~"


@retry(
    stop=stop_after_attempt(20),
    retry=retry_if_exception_type(Exception),
    # reraise=True,  # Reraises if all retries fail
    retry_error_callback=retry_callback  # Returns the value returned here if all retries fail; later, we can patch.
)
def complete_row(row: pd.Series, prefix_column: str) -> str:
    """It seems that this is working well, errors are occurring about every 50 or so when processing 250, and recovering.
    The error isn't really printing out well though. I think it's because of timeouts.
    """
    problem = row["problem"]
    prefix = row[prefix_column]
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


def complete_row_multiple(row: pd.Series, n_completions: int, prefix_column: str) -> list[str]:
    """Generate multiple completions for a single row."""
    completions = []
    for completion_idx in range(n_completions):
        completion = complete_row(row, prefix_column)  # Reuse existing complete_row function
        completions.append(completion)
    return completions

# Changed to async function
async def complete_data(df: pd.DataFrame, n_completions_per_prefix: int, prefix_column: str):
    """
    Generate n completions for each row in the dataframe.
    Returns a new dataframe with completion_idx column and expanded rows.
    """
    rows = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Completing {len(df)} row-solution prefixes with {n_completions_per_prefix} completions each (Sync)"
    ):
        completions = complete_row_multiple(row, n_completions_per_prefix, prefix_column)
        
        # Create n_completions_per_prefix rows for each input row
        for completion_idx, completion in enumerate(completions):
            new_row = row.copy()
            new_row[f"{prefix_column}_completion"] = completion
            new_row["completion_idx"] = completion_idx
            rows.append(new_row)

    return pd.DataFrame(rows)

async def complete_data_parallel(df: pd.DataFrame, n_completions_per_prefix: int, prefix_column: str):
    """
    Generate n completions for each row in the dataframe, using parallel processing.
    Results maintain correct ordering through explicit tracking.
    """
    tasks = []
    for idx, row in df.iterrows():
        for completion_idx in range(n_completions_per_prefix):
            tasks.append({
                'row': row,
                'completion_idx': completion_idx,
                'original_order': len(tasks)  # Track original position
            })
    
    results = []
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        # Submit all tasks
        futures: list[tuple[Future, dict]] = [
            (
                executor.submit(complete_row, task['row'], prefix_column),
                task
            )
            for task in tasks
        ]
        
        # Process results as they complete
        for future, task_info in tqdm(
            futures, 
            desc=f"Completing {len(df)} rows with {n_completions_per_prefix} completions each (Parallel)"
        ):
            try:
                completion = future.result()  # Let complete_row handle retries and timeouts
                
                results.append({
                    'original_order': task_info['original_order'],
                    'row': task_info['row'].copy(),
                    'completion': completion,
                    'completion_idx': task_info['completion_idx']
                })
                
            except Exception as e:
                print(f"WARN: Error for row {task_info['row']['row_id']}, completion {task_info['completion_idx']}: {str(e)}")
                # Append error result to maintain ordering
                results.append({
                    'original_order': task_info['original_order'],
                    'row': task_info['row'].copy(),
                    'completion': f"ERROR: {str(e)}",
                    'completion_idx': task_info['completion_idx']
                })

    # Sort results back to original order
    results.sort(key=lambda x: x['original_order'])
    
    # Convert to DataFrame rows
    rows = []
    for result in results:
        new_row = result['row']
        new_row["completion_idx"] = result['completion_idx']
        new_row[f"{prefix_column}_completion"] = result['completion']
        rows.append(new_row)

    return pd.DataFrame(rows)

# TODO: HEY! REMEMBER TO CHANGE THESE FIRST THINGS BETWEEN RUNS!
async def main():
    is_on_policy = False
    n_completions_per_prefix = 5  # Configure number of completions per prefix

    source_filename = (
        "datasets/cn_k12_math_problems_prefixes_off_policy_command-r-03-2024_191_3_take_01_03_05_07.csv"
    )
    prefix_column = "prefix_take_0.3"
    output_suffix = "take_0.3"  # A bonus suffix to add to output filename, if needed for experiments

    # Load dataframe
    print(f"Loading dataframe from {source_filename}...")
    df = pd.read_csv(source_filename)
    len_df = len(df)
    print(f"Loaded dataframe with {len_df} row-solution prefixes in total!")

    # Find the number of solution_idxs per row_id
    n_prefixes_per_problem = df.groupby('row_id')['solution_idx'].nunique().iloc[0]
    print(f"Number of solution prefixes per row/problem: {n_prefixes_per_problem}")

    # Verify that this number is consistent across all row_ids
    if not (df.groupby('row_id')['solution_idx'].nunique() == n_prefixes_per_problem).all():
        raise ValueError("Inconsistent number of solution_idxs across row_ids -- This shouldn't happen")

    print(f"This should result in {len_df * n_completions_per_prefix} completions in total")

    # Process data in batches, accumulate and concatenate results (with checkpointing)
    # The batch size is the number of the input rows to process, but consider that this results in (bs * n_completions_per_prefix) completions. These completions are generated synchronously, but the verifications are async shot all at once.
    # Consider too that there's currently no sleeping between the sync and async parts.
    # Edit: Now the completions are parallelized using threads
    # Seems like 35 is a safe number...
    bs = 35  # Think: bs * n_completions_per_prefix should be well under 500, the rate limit; Because you're async veriying each batch at once, and you want to account for the need for retries.
    print(f"Processing {len_df} rows in batches of {bs}")
    processed_dfs = []

    # We can naively batch this because each row-solution prefix can be independently completed and verified.
    for i in range(0, len_df, bs):
        batch_df = df.iloc[i : i + bs]
        print(f"Processing batch {i//bs + 1}. Records {i} to {min(i+bs, len_df)} of {len_df}")

        # Generate completions for the batch (sync)
        print(f"Generating completions for batch {i//bs + 1}...")
        completion_batch_df = await complete_data_parallel(batch_df, n_completions_per_prefix, prefix_column)  # Add await here
        print(f"Finished generating completions for batch {i//bs + 1}")

        # Verify the completions for the batch (async)
        print(f"Verifying completions for batch {i//bs + 1}...")
        verified_batch_df = await verify_data(completion_batch_df, prefix_column)
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