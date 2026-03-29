# autoresearch

This repo is an adaptation of the autoresearch idea for a tabular ML project: an LLM agent repeatedly edits an XGBoost training script, runs an experiment, checks whether holdout F1 improved, and keeps or discards the change.

The current project is a LendingClub loan-default classification pipeline built around [`train_xgb.py`](train_xgb.py).

Original autoresearch concept and repo by Andrej Karpathy. See his announcement [here](https://x.com/karpathy/status/2029701092347630069).

## What This Repo Does

The main script:

- loads `data/train.csv`, `data/holdout.csv`, and `data/score.csv`
- cleans and engineers features
- builds a scikit-learn preprocessing pipeline
- runs randomized hyperparameter search for XGBoost
- tunes a classification threshold on the holdout split
- writes search results to `model_results/`
- writes a submission file to `data/kaggle_submissions/`

The primary objective is **holdout F1**. Higher is better.

Holdout AUC is useful context, but F1 is the metric the agent should optimize.

## Current Workflow

This repo is intentionally narrow in scope:

- [`train_xgb.py`](train_xgb.py) is the file the agent edits
- [`program.md`](program.md) contains the instructions for the autonomous experiment loop
- [`data_dictionary.csv`](data_dictionary.csv) is the feature reference the agent should read before making feature engineering changes

The agent loop is simple:

1. run the baseline script
2. change only `train_xgb.py`
3. run another experiment
4. compare holdout F1 against the previous best
5. keep improvements and discard regressions

## Quick Start

Requirements:

- Python 3.10+
- packages from [`requirements.txt`](requirements.txt)
- optional NVIDIA GPU for faster XGBoost training

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_xgb.py
```

The script uses CUDA when available and otherwise falls back to CPU. MPS is not supported by the current code path.

## Files That Matter

```text
train_xgb.py          — active training pipeline; this is the file to modify
program.md            — agent instructions for the autonomous experiment loop
data_dictionary.csv   — feature descriptions for feature engineering decisions
data/train.csv        — training split
data/holdout.csv      — validation/holdout split used for model selection
data/score.csv        — unlabeled scoring split for submission generation
model_results/        — random-search result exports
data/kaggle_submissions/ — generated submission files
utils/emp_title_cleaning.py — helper used by the training script
```

## Output

A typical run prints progress for:

1. loading data
2. cleaning and feature engineering
3. building the preprocessor
4. training random-search candidates
5. generating the submission

At the end of a run, the script writes:

- a CSV of search results in `model_results/`
- a submission CSV in `data/kaggle_submissions/`

The final summary line includes the best parameter set, AUC, F1, and threshold chosen on the holdout set.

## Using An Agent

The repo is designed for an agentic workflow. Point your coding agent at [`program.md`](program.md) and have it operate on this repo directly.

The intended constraints are:

- edit only [`train_xgb.py`](train_xgb.py)
- read [`data_dictionary.csv`](data_dictionary.csv) before substantial feature engineering changes
- track experiments in `results.tsv`
- optimize for holdout F1, not for code complexity on its own

Example prompt:

```text
Look at program.md and start the experiment loop for train_xgb.py.
```

## Notes

- This repo is inspired by the original autoresearch idea, but it is not the original 5-minute LLM training setup.
- Original autoresearch creator: Andrej Karpathy.
- The old `train.py` workflow is not the active path for this project.
- If you update the search space, feature engineering, or threshold logic, keep the script runnable end-to-end and preserve the ability to compare experiments cleanly.

## License

MIT
