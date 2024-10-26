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
To generate straight shot solutions (n solution attempts per problem, regardless of whether they are correct or not), run:
```bash
python generate_straight_shot.py
```

To generate completions for a dataframe with prefixes, run
```bash
python generate_completions.py
```

To generate prefixes (meaning n solutions with incorrect verifications), run
```bash
python generate_prefixes.py
```

View the results of the straight shot solutions by running `python vieww__ss_results.py` (Make sure the file points to the correct CSV file) 

View the results of the completions by running `python view__completions_results.py` (Make sure the file points to the correct CSV file) 

