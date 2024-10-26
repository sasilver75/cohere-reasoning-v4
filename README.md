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
To run the full pipeline to completion, do the following:

Download the dataset from HuggingFace by running `python download_dataset.py`

Generate weak solutions by running `python perturbation/weak_model/run.py`

Generate completions by running `python completion/run.py`

View the results of the completions by running `python completion__view_problem_solution_prefix_trace_completion.py` (Make sure the file points to the correct CSV file) 