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


async def verify_row_completion(row: pd.Series, prefix_results: dict, prefix_columns: list[str]) -> dict[str, VerificationResult]:
    """
    Verify completions for all prefixes in a single row.
    Returns a dict mapping prefix columns to their verification results.
    """
    problem = row["problem"]
    ground_truth_solution = row["ground_truth_solution"]
    row_id = row["row_id"]
    solution_idx = row["solution_idx"]
    completion_idx = row["completion_idx"]
    
    verifications = {}
    for prefix_col in prefix_columns:
        completion = prefix_results[f"{prefix_col}_completion"]
        
        result = await generate_verification(
            row_id=row_id,
            problem=problem,
            ground_truth_solution=ground_truth_solution,
            candidate_solution=completion,
            solution_idx=solution_idx,
            update_request_count=update_request_count,
            strong_verifier_name=verifier_name,
            completion_idx=completion_idx
        )
        verifications[prefix_col] = result
        
    return verifications


async def verify_data(df: pd.DataFrame, prefix_columns: list[str]):
    """
    Generate verifications for candidate solutions in each row of a dataframe.
    """
    coroutines = [
        verify_row_completion(
            row, 
            {
                f"{prefix_col}_completion": row[f"{prefix_col}_completion"]
                for prefix_col in prefix_columns
            },
            prefix_columns
        ) 
        for _, row in df.iterrows()
    ]

    results = []
    for task in atqdm(
        asyncio.as_completed(coroutines),
        total=len(coroutines),
        desc=f"Verifying {len(df)} completions (Async)",
    ):
        result = await task
        results.append(result)

    # Create rows for the verification DataFrame
    verification_rows = []
    for idx, verifications in enumerate(results):
        row = {
            "row_id": df.iloc[idx]["row_id"],
            "solution_idx": df.iloc[idx]["solution_idx"],
            "completion_idx": df.iloc[idx]["completion_idx"],
        }
        
        for prefix_col, result in verifications.items():
            row[f"{prefix_col}_completion_verification_reasoning"] = result.verification_reasoning
            row[f"{prefix_col}_completion_verification"] = result.verification
            
        verification_rows.append(row)

    verification_df = pd.DataFrame(verification_rows)

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
    print(f"ALL RETRIES FAILED: {retry_state}")
    return "~FAILED~"


@retry(
    stop=stop_after_attempt(30),
    retry=retry_if_exception_type(Exception),
    retry_error_callback=retry_callback
)
def complete_row_for_prefix(row: pd.Series, prefix_col: str) -> str:
    """Generate a completion for a single prefix in a row."""
    problem = row["problem"]
    prefix = row[prefix_col]

    user_turn = prompts.COMPLETION_PROMPT_USER.format(problem=problem)
    assistant_turn = prompts.COMPLETION_PROMPT_ASSISTANT.format(prefix=prefix)

    def get_completion():
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
        with ThreadPoolExecutor() as executor:
            future = executor.submit(get_completion)
            completion = future.result(timeout=60)
    except Exception as e:
        print(
            f"Error generating completion for row {row['row_id']} (solution {row['solution_idx']}, prefix {prefix_col}): Type: {type(e).__name__}, Message: {str(e)}, Full repr: {repr(e)}"
        )
        raise e

    return completion.text


def complete_row_all_prefixes(row: pd.Series, prefix_columns: list[str]) -> dict[str, str]:
    """Generate completions for all prefixes in a single row."""
    completions = {}
    for prefix_col in prefix_columns:
        completion = complete_row_for_prefix(row, prefix_col)
        completions[f"{prefix_col}_completion"] = completion
    return completions


async def complete_data_parallel(df: pd.DataFrame, n_completions_per_prefix: int, prefix_columns: list[str]):
    """
    Generate n completions for each row and prefix combination, using parallel processing.
    """
    tasks = []
    for idx, row in df.iterrows():
        for completion_idx in range(n_completions_per_prefix):
            tasks.append({
                'row': row,
                'completion_idx': completion_idx,
                'original_order': len(tasks)
            })
    
    results = []
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures: list[tuple[Future, dict]] = [
            (
                executor.submit(complete_row_all_prefixes, task['row'], prefix_columns),
                task
            )
            for task in tasks
        ]
        
        for future, task_info in tqdm(
            futures,
            desc=f"Completing {len(df)} rows with {n_completions_per_prefix} completions each (Parallel)"
        ):
            try:
                completions = future.result()
                
                results.append({
                    'original_order': task_info['original_order'],
                    'row': task_info['row'].copy(),
                    'completions': completions,
                    'completion_idx': task_info['completion_idx']
                })
                
            except Exception as e:
                print(f"Error for row {task_info['row']['row_id']}, completion {task_info['completion_idx']}: {str(e)}")
                # Append error result
                error_completions = {
                    f"{prefix_col}_completion": f"ERROR: {str(e)}"
                    for prefix_col in prefix_columns
                }
                results.append({
                    'original_order': task_info['original_order'],
                    'row': task_info['row'].copy(),
                    'completions': error_completions,
                    'completion_idx': task_info['completion_idx']
                })

    # Sort results back to original order
    results.sort(key=lambda x: x['original_order'])
    
    # Convert to DataFrame rows
    rows = []
    for result in results:
        new_row = result['row'].copy()
        new_row["completion_idx"] = result['completion_idx']
        for col, completion in result['completions'].items():
            new_row[col] = completion
        rows.append(new_row)

    return pd.DataFrame(rows)


async def main():
    is_on_policy = False
    n_completions_per_prefix = 5

    # Define prefix configurations here
    # prefix_takes = [0.3, 0.5, 0.7]
    # prefix_columns = [f"prefix_take_{p}" for p in prefix_takes]
    prefix_takes = []
    prefix_columns = ["prefix"]  # This is different for the static completions.

    # THINKING: If we want to limit requests to 500/min. For a given input row, we have to do len(prefix_takes)*2 requests (1 for completion, 1 for verification).
    # But we should also think about how often we expect to have to do retries.
    # If we target 300 requests/min without retries, then if our bs=30 and len(prefix_takes)=3, we're at 30*3*2=180 requests/min.
    # AH, but you forgot that we generate the completions (roughly sync) and then the verifications all at once, async. So we really just want to worry about bs*len(prefix_takes)*n_completions_per_prefix
    # ANyways, I'm off to class, so going to run with bs20; Edit: That crashed on the first batch with too many requests. Trying again with 10. That hit 300 RPM max for first batch. let's cross fingers.
    # - 10 rows × 5 completions × 3 prefixes = 150 completion requests
    # - Each completion needs verification = 150 verification requests (total: 300)
    bs = 10

    source_filename = (
        "datasets/cn_k12_math_problems_prefixes_static_191_3.csv"
    )
    output_suffix = "STATIC"

    # Load dataframe
    print(f"Loading dataframe from {source_filename}...")
    df = pd.read_csv(source_filename)
    len_df = len(df)
    print(f"Loaded dataframe with {len_df} row-solution prefixes in total!")

    # Find the number of solution_idxs per row_id
    n_prefixes_per_problem = df.groupby('row_id')['solution_idx'].nunique().iloc[0]
    print(f"Number of solution prefixes per row/problem: {n_prefixes_per_problem}")

    print(f"This should result in {len_df * n_completions_per_prefix * len(prefix_columns)} completions in total, returned in {len_df * n_completions_per_prefix} rows")

    # Process data in batches
    print(f"Processing {len_df} rows in batches of {bs}")
    processed_dfs = []

    # We can naively batch this because each row-solution prefix can be independently completed and verified.
    for i in range(0, len_df, bs):
        batch_df = df.iloc[i : i + bs]
        print(f"Processing batch {i//bs + 1}. Records {i} to {min(i+bs, len_df)} of {len_df}")

        # Generate completions for the batch (sync)
        print(f"Generating completions for batch {i//bs + 1}...")
        completion_batch_df = await complete_data_parallel(
            batch_df, 
            n_completions_per_prefix,
            prefix_columns
        )
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
    output_filename = (
        f"datasets/cn_k12_math_problems_completions_{completer_name}"
        f"_{verified_df['row_id'].nunique()}_{n_prefixes_per_problem}_{n_completions_per_prefix}"
        f"_{'ON' if is_on_policy else 'OFF'}_take_{"_".join(str(take) for take in prefix_takes)}_COMBINED{"_"+ output_suffix if output_suffix else ''}.csv"
    )

    print(f"Saving results to {output_filename}...")
    verified_df.to_csv(output_filename, index=False)
    print(f"Saved results to {output_filename}!")

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())