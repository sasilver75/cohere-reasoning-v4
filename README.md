# Inflight, Intrinsic Self-Correction (v3)

## Installation

Make sure you have [Pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) installed with the correct version (3.12.6) of Python available.
```bash
pyenv install 3.12.6
```

Now, with this repository as your current working directory,create your virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

To use the Cohere API, you'll need an API key. Create an `.env` file with the following:
```bash
COHERE_API_KEY=your_actual_api_key_here
```
This environment variable file is git-ignored, so your precious credentials won't be checked into the git repository.


## Processing data

Descriptions of scripts:

`generate_straight_shot.py`
- For a given completer (either strong or weak), generates N (eg 5) solutions attempts per Numina problem (# configurable).
- For each solution, verifies it using the strong verifier.
- Resulting dataframe has columns: row_id, problem, ground_truth_solution, solution_idx, solution, verification_reasoning, verification
    - `row_id` is the Numina problem ID
    - `problem` is the Numina problem text
    - `ground_truth_solution` is the Numina problem's ground truth solution
    - `solution_idx` is the index of the solution attempt, between 0 and N-1
    - `solution` is the candidate solution generated by the completer
    - `verification_reasoning` is the strong verifier's reasoning for the verification dispensed in `verification`
    - `verification` is the verification result (either correct or incorrect), based on whether the candidate solution final solution matches the ground truth final solution.
- Produces csv files named as (eg) `datasets/cn_k12_math_problems_ss_command-r-plus-08-2024_5_2.csv`
    - `ss` denotes "straight shot"
    - `command-r-plus-08-2024` denotes the completer model used
    - `5` denotes the number of Numina problems
    - `2` denotes the number of solutions generated per problem
- From this saved data, we'll be able to determine the difficulty level of each Numina problem, according to the completer used, by considering the percentage of solution attempts that were verified as correct.


`generate_solvable_incorrect_with_extracted_prefixes.py`
- Given the results of `generate_straight_shot.py`, does a number of useful things.
    - Determines the collection of solvable problems (i.e. those with success rates of (eg) 20-80%, across solution attempts).
    - Saves this list of solveable row IDs to a text file, named as (eg) `datasets/cn_k12_math_problems_sri_command-r-plus-08-2024_2.txt`.
        - `sri` denotes "solveable problem row IDs"
        - `command-r-plus-08-2-24` denotes the completer model used in the previous step; this should not be changed in this step.
        - `2` denotes the number of solvable problems.
    - Filters the original dataframe from `generate_straight_shot.py` to only include the solvable problems, and only the solutions for these problems that were determined to be _incorrect_ by the strong verifier.
    - For this subset, extracts prefixes and adds as a new column `prefix`.
    - Saves this dataframe to a csv file, named as (eg) `datasets/cn_k12_math_problems_srip_command-r-plus-08-2024_2_2.csv`.
        - `srip` denotes "solveable problem dataframe with extracted prefixes"
        - `2` denotes the number of solvable problems.
        - `2` denotes the *total* number of incorrect solutions with prefixes generated across all solvable problems. There may be differing numbers of incorrect solutions per solvable problem.
        - This file is saved because it's useful downstream for the `generate_prefixes_remaining_on_policy.py` script, which will generate any remaining required prefixes for the solvable problems. Harveting and saving these on-policy prefixes from the (eg) strong completer straight-shot solutions is just a way of saving compute credits.
- Resulting dataframe has columns: row_id, problem, ground_truth_solution, solution_idx, solution, verification_reasoning, verification, prefix
    - `prefix` is a leading portion (currently, first 30% of words) of the incorrect solution, which we hope constitutes some sort of reasoning perturbation that will be used to assess self-correction abilities of the completer, downstream.


`generate_prefixes_remaining_on_policy.py`
- Given the results of `generate_solvable_incorrect_with_extracted_prefixes.py` (specifically the dataframe, rather than the text file of solveable row IDs), generates any remaining required incorrect solution prefixes for the solvable problems.
- Doesn't add any columns to the dataframe, but produces a new dataframe with additional rows (padding the number of incorrect prefixes to the total number required for each solvable problem).
- Saves dataframe to a csv file, named as (eg) `datasets/cn_k12_math_problems_prefixes_on_policy_command-r-plus-08-2024_100_5.csv`
    - `prefixes_on_policy` denotes that these are on-policy prefixes
    - `command-r-plus-08-2024` denotes the completer model used
    - `100` denotes the number of solvable problems
    - `5` denotes the number of incorrect prefixes per solvable problem.

`generate_prefixes_off_policy.py`
- TBD


`generate_completions.py`
- Given the results of either `generate_prefixes_remaining_on_policy.py` or `generate_prefixes_off_policy.py`, generates a configurable number of completions for each row-solution-prefix.
    - While increasing the number of prefixes per problems (hopefully) increases the number of possible perturbations that we're able to test, increasing the number of completions per prefix gives us a better estimate (when results are averaged across verified prefix-completions) of the true self-correction ability of the completer, for a given prefix.
- Resulting dataframe has columns: row_id, problem, ground_truth_solution, solution_idx, candidate_solution, verification_reasoning, verification, prefix, completion, completion_idx, completion_verification_reasoning, completion_verification
- Produces csv files named as (eg) `datasets/cn_k12_math_problems_completions_command-r-plus-08-2024_2_5_2_ON.csv`
    - `completions` denotes that these are completions
    - `command-r-plus-08-2024` denotes the completer model used
    - `2` denotes the number of solvable problems (supplied in input DF)
    - `5` denotes the number of incorrect prefixes per solvable problem (supplied in input DF)
    - `2` denotes the number of completions per prefix (generated in this script; multiplying `2` * `5` * `2` = `20` completions in total)
    - `ON` denotes that these completions are on-policy
- Now the unit of analysis can be as granular as (row_id, solution_idx, completion_idx) triplets. The output dataframe CSV is ready for downstream analysis and plotting.

## Viewing results

View the results of the straight shot solutions by running `python vieww__ss_results.py` (Make sure the file points to the correct CSV file) 

View the results of the completions by running `python view__completions_results.py` (Make sure the file points to the correct CSV file) 

