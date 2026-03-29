# autoresearch

This is an experiment to have the LLM do its own research on improving `train_xgb.py`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar29`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `data_dictionary.csv` — feature meanings. Look at this before making feature engineering changes.
   - `train_xgb.py` — the file you modify. Data cleaning, preprocessing, XGBoost random search, threshold tuning, and final retraining.
4. **Verify data exists**: Check that `data/train.csv`, `data/holdout.csv`, and `data/score.csv` exist. If not, tell the human the data files are missing.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single machine by executing the XGBoost training script. You launch it simply as: `python train_xgb.py`.

**What you CAN do:**
- Modify `train_xgb.py` — this is the only file you edit. Everything is fair game: feature engineering, preprocessing, hyperparameters, threshold tuning, the number of random search iterations, retraining the best model, etc.

**What you CANNOT do:**
- Modify `data_dictionary.csv` or the raw csv data files. They are read-only reference/input files.
- Install new packages or add dependencies. You can only use what's already in the repo.
- Modify other Python files. `train_xgb.py` is the ground truth file you change.

**The goal is simple: get the highest holdout F1.** Since `train_xgb.py` already does x iterations of random search and retrains the best model, you don't need to redesign the whole workflow — improve it. Everything is fair game: change the feature engineering, the preprocessing, the hyperparameters, the threshold search, and the random search space. The only constraint is that the code runs without crashing and completes end-to-end.

**Runtime** is a soft constraint. Some increase is acceptable for meaningful F1 gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 F1 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 F1 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints summary lines like this:

```
  Results -> model_results/03-29_xgb_random_search_results.csv
  Best params={...}  AUC=0.7050  F1=0.7400  thr=0.370
    Saved -> data/kaggle_submissions/03-29_xgb_best_f10.7400_thr0.370.csv
```

You can extract the key metric from the log file:

```
grep "Best params=" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	holdout_f1	holdout_auc	status	description
```

1. git commit hash (short, 7 chars)
2. holdout F1 achieved (e.g. 0.740000) — use 0.000000 for crashes
3. holdout AUC achieved (e.g. 0.705000) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	holdout_f1	holdout_auc	status	description
a1b2c3d	0.732800	0.691200	keep	baseline
b2c3d4e	0.740000	0.705000	keep	add missingness indicators
c3d4e5f	0.736500	0.704200	discard	raise max_depth
d4e5f6g	0.000000	0.000000	crash	bad categorical handling
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar29` or `autoresearch/mar29-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train_xgb.py` with an experimental idea by directly hacking the code. Before making feature engineering changes, look at `data_dictionary.csv`.
3. git commit
4. Run the experiment: `python train_xgb.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "Best params=" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If holdout F1 improved (higher), you "advance" the branch, keeping the git commit
9. If holdout F1 is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take a reasonable amount of time for the current `N_RANDOM_SAMPLES` and search space. If a run exceeds the expected runtime by a lot, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical feature engineering or search changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment length depends on the current search size and hardware, but the user should wake up to a long series of completed experiments, all completed by you while they slept.
