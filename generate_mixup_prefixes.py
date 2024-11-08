import pandas as pd
import numpy as np
from tqdm import tqdm

def main(input_prefix_dataframe_filepath: str, output_prefix_dataframe_filepath: str):
    """
    This can be a synchronous function; I think it's basically just going to be mixing and matches collections of string prefixes.    
    """
    # Read the input dataframe
    print(f"Reading dataframe from {input_prefix_dataframe_filepath}")
    df = pd.read_csv(input_prefix_dataframe_filepath)
    print(f"Read dataframe with {len(df)} rows; {df["row_id"].nunique()} unique problems")
    
    # Get unique row_ids (ignoring solution_idx)
    unique_row_ids = df['row_id'].unique()
    
    # Create a mapping of row_ids to their shuffled partners (Different treatment if there's an odd number, since it has to be paired with some random partner)
    shuffled_pairs = np.random.permutation(unique_row_ids)
    if len(shuffled_pairs) % 2 == 1:
        last_id = shuffled_pairs[-1]
        random_partner = np.random.choice(shuffled_pairs[:-1])
        row_id_pairs = list(zip(shuffled_pairs[:-1:2], shuffled_pairs[1:-1:2]))
        row_id_pairs.append((last_id, random_partner))
    else:
        row_id_pairs = list(zip(shuffled_pairs[::2], shuffled_pairs[1::2]))
    
    # Prefix columns to shuffle
    prefix_columns = ['prefix_take_0.1', 'prefix_take_0.3', 'prefix_take_0.5', 'prefix_take_0.7']
    
    # Create a copy of the dataframe to modify
    df_mixed = df.copy()

    print(f"Beginning swapping of {len(row_id_pairs)} pairs of rows (and their associated solutions' prefixes) for columns {prefix_columns}")
    
    # Perform the prefix swapping
    # Note that it's a rare occurrence that two row_ids are going to have an uneven number of solutions. This is a result when one of the problems is so easy that we weren't able to generate the desired number of solutions
    # before running out of generate-verify retries. In that case, we're just going to not swap the remaining item(s). So if row_1 has A,B,C and row_2 has A,B, we'll swap A/Bs, but leave the C unswapped. This is a small amount of noise.
    # Perform the prefix swapping
    for id1, id2 in tqdm(row_id_pairs, desc="Swapping prefixes", total=len(row_id_pairs)):
        # Get all rows for both row_ids (including all solution_idxs)
        rows1 = df[df['row_id'] == id1].index
        rows2 = df[df['row_id'] == id2].index
        
        # Find how many solutions we can swap (minimum of the two)
        n_solutions = min(len(rows1), len(rows2))
        
        # Take the first n_solutions rows from each row_id
        swap_rows1 = rows1[:n_solutions]
        swap_rows2 = rows2[:n_solutions]
        
        # Store original prefixes
        temp_prefixes1 = df.loc[swap_rows1, prefix_columns].values
        temp_prefixes2 = df.loc[swap_rows2, prefix_columns].values
        
        # Swap prefixes
        df_mixed.loc[swap_rows1, prefix_columns] = temp_prefixes2
        df_mixed.loc[swap_rows2, prefix_columns] = temp_prefixes1
    
    # Save the mixed dataset
    df_mixed.to_csv(output_prefix_dataframe_filepath, index=False)
    print(f"Saved mixed dataframe to {output_prefix_dataframe_filepath}")

if __name__ == "__main__":
    # The path to the on-policy prefixes we want to MixUp
    input_prefix_dataframe_filepath = "datasets/cn_k12_math_problems_prefixes_on_policy_command-r-plus-08-2024_191_3_take_0.1_0.3_0.5_0.7.csv"
    # Determine output filepath as input filepath with MIXUP suffix
    output_prefix_dataframe_filepath = input_prefix_dataframe_filepath.replace(".csv", "_MIXUP.csv")
    
    main(input_prefix_dataframe_filepath, output_prefix_dataframe_filepath)